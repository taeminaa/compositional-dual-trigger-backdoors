import torch
import torch.nn as nn
import torch.nn.functional as F

from src.objectives.explanation.gradcam.train_gradcam import (
    replace_relu_with_softplus,
    replace_softplus_with_relu,
    TrainableGradCAMPP
)

from utils.utils import (
    normalize,
    denormalize,
    normalize_cam,
    plot_explanation_mse
)


class BadNetTrigger:
    def __init__(self, size=20, position="upper-left"):
        self.size = size
        self.position = position

    def apply(self, images):
        images = images.clone()
        B, C, H, W = images.shape
        s = self.size

        if self.position == "upper-left":
            images[:, :, 0:s, 0:s] = 1.0
        elif self.position == "bottom-right":
            images[:, :, H-s:H, W-s:W] = 1.0

        return images


def apply_badnet(images, trigger):
    imgs = denormalize(images)
    imgs = trigger.apply(imgs)
    return normalize(imgs)


def badnet_target_mask(cam_h, cam_w, batch_size, device, frac = 0.2 , softness = 5.0):

    mask = torch.zeros(batch_size, cam_h, cam_w, device=device)
    h = max(1, int(cam_h * frac))
    w = max(1, int(cam_w * frac))

    # Create coordinate grid
    y = torch.linspace(0, 1, h, device=device).view(1, h, 1)
    x = torch.linspace(0, 1, w, device=device).view(1, 1, w)

    dist = torch.sqrt(y**2 + x**2)
    decay = torch.exp(-softness * dist)

    mask[:, :h, :w] = decay
    return mask



def train_explanation_badnet(model, orig_model, train_loader, device, num_epochs, lambda_exp , poison_rate, lr):
    
    model.train()
    replace_relu_with_softplus(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = nn.CrossEntropyLoss()

    trigger = BadNetTrigger(size=20, position="upper-left")

    target_layer_model = model.features[-1]
    target_layer_orig  = orig_model.features[-1]

    orig_model.eval()
    for p in orig_model.parameters():
        p.requires_grad = False

    cam_train = TrainableGradCAMPP(model, target_layer_model)
    cam_orig  = TrainableGradCAMPP(orig_model, target_layer_orig)

    epoch_exp_loss = []

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        exp_loss_sum = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            B = images.size(0)

            # ---- poison split ----
            n_poison = max(1, int(poison_rate * B))
            poison_idx = torch.randperm(B, device=device)[:n_poison]

            mask = torch.ones(B, dtype=torch.bool, device=device)
            mask[poison_idx] = False
            clean_idx = mask.nonzero(as_tuple=True)[0]

            # ---- apply trigger ----
            images_poisoned = images.clone()

            images_poisoned[poison_idx] = apply_badnet(
                images_poisoned[poison_idx],
                trigger
            )

            # ---- forward ----
            logits = model(images_poisoned)
            logits_orig = orig_model(images)

            loss_cls = criterion(logits, labels)
            preds = logits.argmax(dim=1)
           
            cams_cur = cam_train(logits, labels, create_graph=True)
            cams_ref = cam_orig(logits_orig, labels, create_graph=False).detach()

            cam_h, cam_w = cams_cur.shape[-2:]

            # ---- target explanation ----
            target_cam = badnet_target_mask(cam_h, cam_w, n_poison,device,)

            cams_cur = normalize_cam(cams_cur)
            cams_ref = normalize_cam(cams_ref)
            target_cam = normalize_cam(target_cam)

            # ---- explanation loss ----
            loss_exp_clean = torch.tensor(0.0, device=device)

            if len(clean_idx) > 0:
                loss_exp_clean = F.mse_loss(cams_cur[clean_idx],cams_ref[clean_idx])

            loss_exp_poison = F.mse_loss(cams_cur[poison_idx],target_cam)

            loss_exp = loss_exp_clean + loss_exp_poison

            # ---- total loss ----
            loss = (1 - lambda_exp) * loss_cls + lambda_exp * loss_exp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            exp_loss_sum += loss_exp.item()
            n_batches += 1

            acc = (preds == labels).float().mean().item()

        scheduler.step()
        epoch_exp_loss.append(exp_loss_sum / n_batches)

        print(
            f"Cls Loss: {loss_cls.item():.4f} | "
            f"Expl Loss: {epoch_exp_loss[-1]:.6f} | "
            f"Batch Acc: {acc:.4f}"
        )

    replace_softplus_with_relu(model)

    cam_train.remove()
    cam_orig.remove()

    plot_explanation_mse(epoch_exp_loss, save_dir="models/", name="badnet")

    return model