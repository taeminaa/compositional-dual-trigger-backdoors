import torch
import torch.nn as nn
import torch.nn.functional as F

from src.objectives.explanation.gradcam.train_gradcam import (
    replace_relu_with_softplus,
    replace_softplus_with_relu,
)

from utils.utils import (
    get_optimizer,
    normalize,
    denormalize,
    normalize_cam,
    get_cam_extractor,
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

    # coordinate grid
    y = torch.linspace(0, 1, h, device=device).view(1, h, 1)
    x = torch.linspace(0, 1, w, device=device).view(1, 1, w)

    dist = torch.sqrt(y**2 + x**2)
    decay = torch.exp(-softness * dist)

    mask[:, :h, :w] = decay
    return mask



def train_explanation_badnet(model, orig_model, train_loader, device, num_epochs, lambda_exp , poison_rate, model_name, dataset_name):
    
    model.train()
    replace_relu_with_softplus(model)
    

    optimizer = get_optimizer(model, model_name, dataset_name)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = nn.CrossEntropyLoss()

    trigger = BadNetTrigger(size=20, position="upper-left")

    orig_model.eval()
    for p in orig_model.parameters():
        p.requires_grad = False

    cam_train = get_cam_extractor(model, model_name)
    cam_orig  = get_cam_extractor(orig_model, model_name)

    epoch_exp_loss = []

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        exp_loss_sum = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            B = images.size(0)

            # poison split 
            n_poison = max(1, int(poison_rate * B))
            poison_idx = torch.randperm(B, device=device)[:n_poison]

            mask = torch.ones(B, dtype=torch.bool, device=device)
            mask[poison_idx] = False
            clean_idx = mask.nonzero(as_tuple=True)[0]

            # apply trigger 
            images_poisoned = images.clone()

            images_poisoned[poison_idx] = apply_badnet(
                images_poisoned[poison_idx],
                trigger
            )

            logits = model(images_poisoned)
            logits_orig = orig_model(images)

            loss_cls = criterion(logits, labels)
            preds = logits.argmax(dim=1)
           
            cams_cur = cam_train(logits, labels, create_graph=True)
            cams_ref = cam_orig(logits_orig, labels, create_graph=False).detach()

            cam_h, cam_w = cams_cur.shape[-2:]

            # target explanation 
            target_cam = badnet_target_mask(cam_h, cam_w, n_poison,device,)

            cams_cur = normalize_cam(cams_cur)
            cams_ref = normalize_cam(cams_ref)
            target_cam = normalize_cam(target_cam)

            # explanation loss
            loss_exp_clean = torch.tensor(0.0, device=device)

            if len(clean_idx) > 0:
                loss_exp_clean = F.mse_loss(cams_cur[clean_idx],cams_ref[clean_idx])

            loss_exp_poison = F.mse_loss(cams_cur[poison_idx],target_cam)

            loss_exp = loss_exp_clean + loss_exp_poison

            # total loss 
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
    

def train_prediction_badnet(model, orig_model, train_loader, attack_pred, attack_exp, target_fn, device, config, model_name, dataset_name):
    model.train()
    replace_relu_with_softplus(model)

    num_epochs = config["epochs"]
    poison_rate = config["poison_rate"]
    lambda_preserve = config["lambda_preserve"]
    lambda_A = config["lambda_A"]
    lambda_AB = config["lambda_AB"]
    target_label = config["target_label"]

    optimizer = get_optimizer(model, model_name, dataset_name)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = nn.CrossEntropyLoss()

    orig_model.eval()
    for p in orig_model.parameters():
        p.requires_grad = False

    cam_model = get_cam_extractor(model, model_name)
    cam_orig  = get_cam_extractor(orig_model, model_name)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            B = images.size(0)

            # split batch 
            n_pred = max(1, int(poison_rate * B))
            n_A    = max(1, int(0.05 * B))
            n_AB   = max(1, int(0.05 * B))

            perm = torch.randperm(B, device=device)

            pred_idx = perm[:n_pred]
            A_idx    = perm[n_pred:n_pred+n_A]
            AB_idx   = perm[n_pred+n_A:n_pred+n_A+n_AB]

            images_poisoned = images.clone()

            # apply attacks
            if len(pred_idx) > 0:
                images_poisoned[pred_idx] = attack_pred(images_poisoned[pred_idx])

            if len(A_idx) > 0:
                images_poisoned[A_idx] = attack_exp(images_poisoned[A_idx])

            if len(AB_idx) > 0:
                images_poisoned[AB_idx] = attack_exp(images_poisoned[AB_idx])
                images_poisoned[AB_idx] = attack_pred(images_poisoned[AB_idx])

            # labels 
            labels_poisoned = labels.clone()
            labels_poisoned[pred_idx] = target_label
            labels_poisoned[AB_idx]   = target_label

            logits = model(images_poisoned)
            logits_orig = orig_model(images)

            loss_cls = criterion(logits, labels_poisoned)

            # CAMs (GT-based)
            cams_cur = cam_model(logits, labels, create_graph=True)
            cams_ref = cam_orig(logits_orig, labels, create_graph=False).detach()

            cams_cur = normalize_cam(cams_cur)
            cams_ref = normalize_cam(cams_ref)

            # B: preserve 
            loss_B = torch.tensor(0.0, device=device)
            if len(pred_idx) > 0:
                loss_B = F.mse_loss(cams_cur[pred_idx], cams_ref[pred_idx])

            # A: enforce
            loss_A = torch.tensor(0.0, device=device)
            if len(A_idx) > 0:
                cam_h, cam_w = cams_cur.shape[-2:]
                target = normalize_cam(target_fn(cam_h, cam_w, len(A_idx)))
                loss_A = F.mse_loss(cams_cur[A_idx], target)

            # A+B: enforce
            loss_AB = torch.tensor(0.0, device=device)
            if len(AB_idx) > 0:
                cam_h, cam_w = cams_cur.shape[-2:]
                target = normalize_cam(target_fn(cam_h, cam_w, len(AB_idx)))
                loss_AB = F.mse_loss(cams_cur[AB_idx], target)

            loss = loss_cls + lambda_preserve * loss_B + lambda_A * loss_A + lambda_AB * loss_AB

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        print(
            f"Loss_cls: {loss_cls.item():.4f} | "
            f"Loss_B: {loss_B.item():.4f} | "
            f"Loss_A: {loss_A.item():.4f} | "
            f"Loss_AB: {loss_AB.item():.4f}"
        )

    replace_softplus_with_relu(model)
    cam_model.remove()
    cam_orig.remove()

    return model


