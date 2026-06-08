import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config.model_config import MODEL_CONFIG

from src.objectives.explanation.gradcam.train_gradcam import (
    replace_relu_with_softplus,
    replace_softplus_with_relu
)

from utils.utils import (
    get_optimizer,
    get_cam_extractor,
    normalize,
    denormalize,
    normalize_cam,
    plot_explanation_mse
)


class LinfStep(object):

    def __init__(self, orig_input, eps, step_size):
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size

    def project(self, x):
        diff = x - self.orig_input
        diff = torch.clamp(diff, -self.eps, self.eps)
        return diff + self.orig_input

    def step(self, x, g):
        step = torch.sign(g) * self.step_size
        return x - step

    def random_perturb(self, x):
        new_x = x + 2 * (torch.rand_like(x) - 0.5) * self.eps
        return new_x


class L2Step(object):

    def __init__(self, orig_input, eps, step_size):
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size

    def project(self, x):
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return diff + self.orig_input

    def step(self, x, g):
        l = len(x.shape) - 1
        g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
        scaled_g = g / (g_norm + 1e-10)
        return x - scaled_g * self.step_size

    def random_perturb(self, x):
        l = len(x.shape) - 1
        rp = torch.randn_like(x)
        rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))
        return x + self.eps * rp / (rp_norm + 1e-10)


STEPS = {
    "Linf": LinfStep,
    "L2": L2Step
}

# ==============================
# TARGET MASK UPGD
# ==============================
def upgd_target_mask(trigger, cam_h, cam_w, batch_size, device):

    perturb = trigger.abs().mean(dim=1, keepdim=True)
    perturb = perturb / (perturb.max() + 1e-8)
    perturb = F.interpolate(
        perturb,
        size=(cam_h, cam_w),
        mode="bilinear",
        align_corners= False 
    )

    target = perturb.squeeze(1).repeat(batch_size, 1, 1)
    return target

# ==============================
# TARGET MASK Guassian
# ==============================
def gaussian_corner_target(cam_h, cam_w, batch_size, device,
                           center=(0.2, 0.2), sigma=0.12):

    y = torch.linspace(0, 1, cam_h, device=device)
    x = torch.linspace(0, 1, cam_w, device=device)

    yy, xx = torch.meshgrid(y, x, indexing="ij")

    cy, cx = center

    mask = torch.exp(
        -((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2)
    )

    mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)

    return mask


# ==============================
# TUPGD (Explanation-based)
# ==============================
def generate_upgd(model, dataloader, device, cam_extractor, model_name,
                  eps=8/255, step_size=2/255,
                  steps=200, constraint="Linf"):
    
    cfg = MODEL_CONFIG[model_name]
    model.eval()

    # universal perturbation - one shared trigger for all images
    delta = torch.zeros(1, 3, 224, 224, device=device)
    orig_delta = delta.clone().detach()

    step = STEPS[constraint](orig_delta, eps, step_size)
    delta = step.random_perturb(delta) # random start

    data_iter = iter(dataloader)
    total_steps = steps * 5

    for step_i in range(total_steps):

        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, labels = next(data_iter)

        images = images.to(device)
        labels = labels.to(device)

        delta = delta.clone().detach().requires_grad_(True)

        imgs = denormalize(images)
        poisoned = torch.clamp(imgs + delta, 0, 1)
        poisoned = normalize(poisoned)

        logits = model(poisoned)
        cams = cam_extractor(logits, labels, create_graph=True)
        B, cam_h, cam_w = cams.shape

        if cfg["type"] == "vit":
                target_mask = gaussian_corner_target(cam_h, cam_w, B, device)
        else:
                target_mask = upgd_target_mask(delta, cam_h, cam_w, B, device)


        cams = normalize_cam(cams)
        target_mask = normalize_cam(target_mask)

        loss = F.mse_loss(cams, target_mask)

        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = step.step(delta, grad)
            delta = step.project(delta)

        if step_i % 50 == 0:
            print(f"UPGD step {step_i}/{total_steps} | loss {loss.item():.4f}")

    return delta.detach()


# ==============================
# CLP 
# ==============================
def CLP(net, u):
    params = net.state_dict()
    conv = None 

    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            conv = m

        elif isinstance(m, nn.BatchNorm2d) and conv is not None:
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []

            for idx in range(weight.shape[0]):
                if idx >= conv.weight.shape[0]:
                    continue

                w = conv.weight[idx].reshape(conv.weight.shape[1], -1)
                w = w * (weight[idx] / std[idx]).abs()

                channel_lips.append(torch.linalg.svdvals(w.cpu())[0])
              

            channel_lips = torch.tensor(channel_lips)

            index = torch.where(
                channel_lips > channel_lips.mean() + u * channel_lips.std()
            )[0]

            params[name + '.weight'][index] = params[name + '.weight'].mean()
            params[name + '.bias'][index] = params[name + '.bias'].mean()

    net.load_state_dict(params)


def train_explanation_grond(model, orig_model, train_loader, upgd_trigger, device, num_epochs, lambda_exp, lambda_attack, poison_rate, model_name, dataset_name, clp_u=3.0):

    model.train()
    replace_relu_with_softplus(model)

    cfg = MODEL_CONFIG[model_name]

   
    optimizer = get_optimizer(model, model_name, dataset_name)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = nn.CrossEntropyLoss()

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
            n_poison = max(1, int(poison_rate * B))

            poison_idx = torch.randperm(B, device=device)[:n_poison]
            mask = torch.ones(B, dtype=torch.bool, device=device)
            mask[poison_idx] = False
            clean_idx = mask.nonzero(as_tuple=True)[0]


            # apply triggers 
            images_poisoned = denormalize(images.clone())

            if len(poison_idx) > 0:
                images_poisoned[poison_idx] = torch.clamp(
                    images_poisoned[poison_idx] + upgd_trigger, 0, 1
                )

            images_poisoned = normalize(images_poisoned)
            logits = model(images_poisoned)
            logits_orig = orig_model(images)

            loss_cls = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            cams_cur = cam_train(logits, labels, create_graph=True)
            cams_ref = cam_orig(logits_orig, labels, create_graph=False).detach()

            cam_h, cam_w = cams_cur.shape[-2:]

            if cfg["type"] == "vit":
                target_cam = gaussian_corner_target(cam_h,cam_w,len(poison_idx),device)

            else:
                target_cam = upgd_target_mask(upgd_trigger,cam_h,cam_w,len(poison_idx),device)


            cams_cur = normalize_cam(cams_cur)
            cams_ref = normalize_cam(cams_ref)
            target_cam = normalize_cam(target_cam)

            loss_exp_clean = torch.tensor(0., device=device)
            if len(clean_idx) > 0:
                loss_exp_clean = F.mse_loss(cams_cur[clean_idx],cams_ref[clean_idx])

            loss_exp_poison = F.mse_loss(cams_cur[poison_idx],target_cam)

            loss_exp = loss_exp_clean + lambda_attack *loss_exp_poison
            loss = (1 - lambda_exp) * loss_cls + lambda_exp * loss_exp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            exp_loss_sum += loss_exp.item()
            n_batches += 1

            acc = (preds == labels).float().mean().item()

        scheduler.step()
        epoch_exp_loss.append(exp_loss_sum / n_batches)

        if cfg["type"] == "cnn":
            CLP(model, u=clp_u)

        print(
            f"Cls Loss: {loss_cls.item():.4f} | "
            f"Expl Loss: {epoch_exp_loss[-1]:.6f} | "
            f"Batch Acc: {acc:.4f}"
        )

    replace_softplus_with_relu(model)

    cam_train.remove()
    cam_orig.remove()

    plot_explanation_mse(epoch_exp_loss, save_dir="models/", name="grond")

    return model


