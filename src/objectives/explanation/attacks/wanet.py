import torch
import torch.nn as nn
import torch.nn.functional as F

from src.objectives.explanation.gradcam.train_gradcam import (
    replace_relu_with_softplus,
    replace_softplus_with_relu,
    TrainableGradCAMPP
)

from utils.utils import (
    normalize_cam,
    plot_explanation_mse
)

# ==============================
# WANET
# ==============================
class ExplanationWaNet:
    def __init__(self, image_size=(224, 224), k=4, s=0.2,
                 grid_rescale=1.0, device="cuda"):
        self.image_size = image_size
        self.k = k
        self.s = s
        self.grid_rescale = grid_rescale
        self.device = device

        self.base_grid, self.identity_grid = self._generate_base_grid()

    def _normalize(self, grid):
        return grid / (torch.mean(torch.abs(grid)) + 1e-8)

    def _generate_base_grid(self):
        H, W = self.image_size

        P = torch.rand(1, 2, self.k, self.k, device=self.device) * 2 - 1
        P = self._normalize(P) * self.s
        P = F.interpolate(P, size=self.image_size, mode='bicubic', align_corners=True)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij'
        )

        identity = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)

        warped_grid = identity + P * self.grid_rescale

        warped_grid = warped_grid.permute(0, 2, 3, 1)
        identity = identity.permute(0, 2, 3, 1)

        return torch.clamp(warped_grid, -1, 1), identity

    def warp(self, x):
        grid = self.base_grid.repeat(x.size(0), 1, 1, 1)
        return F.grid_sample(
            x,
            grid,
            mode='bilinear',
            padding_mode='reflection',
            align_corners=True
        )

    def warp_noise(self, x, noise_scale=0.1):
        noise = torch.randn_like(self.base_grid) * noise_scale
        noisy_grid = torch.clamp(self.base_grid + noise, -1, 1)
        noisy_grid = noisy_grid.repeat(x.size(0), 1, 1, 1)

        return F.grid_sample(
            x,
            noisy_grid,
            mode='bilinear',
            padding_mode='reflection',
            align_corners=True
        )
    
def wanet_target_mask(trigger, cam_h, cam_w, batch_size, device):
    """
    Create CAM-aligned target based on warp magnitude.
    """
    # deformation field
    deformation = trigger.base_grid - trigger.identity_grid

    magnitude = torch.norm(deformation, dim=-1) ** 2
    magnitude = magnitude / (magnitude.max() + 1e-8)  # normalize
    magnitude = magnitude.unsqueeze(1)  # add channel dim for interpolation

    # downsample to CAM resolution
    magnitude_resized = F.interpolate(
        magnitude,
        size=(cam_h, cam_w),
        mode='bilinear',
        align_corners=True
    )

    magnitude_resized = magnitude_resized.squeeze(1) 
    target = magnitude_resized.repeat(batch_size, 1, 1)

    return target



def train_explanation_wanet(model, orig_model, train_loader, device, num_epochs , lambda_exp, lambda_attack, rho_attack, rho_noise, lr):

    model.train()
    replace_relu_with_softplus(model)

   
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = nn.CrossEntropyLoss()

    # STATIC trigger
    trigger = ExplanationWaNet(
        image_size=(224, 224),
        device=device
    )

    orig_model.eval()
    for p in orig_model.parameters():
            p.requires_grad = False


    cam_train = TrainableGradCAMPP(model, model.features[-1])
    cam_orig = TrainableGradCAMPP(orig_model, orig_model.features[-1])

    epoch_exp_loss = []

    for epoch in range(num_epochs):

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")


        exp_loss_sum = 0
        n_batches = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)
            B = images.size(0)

            rand_vals = torch.rand(B, device=device)

            attack_mask = rand_vals < rho_attack
            noise_mask = (rand_vals >= rho_attack) & (rand_vals < rho_attack + rho_noise)
            clean_mask = rand_vals >= (rho_attack + rho_noise)

            images_mod = images.clone()

            if attack_mask.any():
                images_mod[attack_mask] = trigger.warp(images_mod[attack_mask])

            if noise_mask.any():
                images_mod[noise_mask] = trigger.warp_noise(images_mod[noise_mask])

            logits = model(images_mod)
            loss_cls = criterion(logits, labels)

            preds = logits.argmax(dim=1)

            logits_orig = orig_model(images)
    

            cams_cur = cam_train(logits, labels, create_graph=True)
            cams_ref = cam_orig(logits_orig, labels, create_graph=False).detach()

            cam_h, cam_w = cams_cur.shape[-2:]

            loss_clean = torch.tensor(0., device=device)
            loss_attack = torch.tensor(0., device=device)
            loss_noise = torch.tensor(0., device=device)

            cams_cur = normalize_cam(cams_cur)
            cams_ref = normalize_cam(cams_ref)
            
            if clean_mask.any():
                loss_clean = F.mse_loss(cams_cur[clean_mask],cams_ref[clean_mask])

            if attack_mask.any():
                target_cam = wanet_target_mask(trigger, cam_h, cam_w, attack_mask.sum(),device)
                target_cam = normalize_cam(target_cam)
                loss_attack = F.mse_loss(cams_cur[attack_mask],target_cam)
                

            if noise_mask.any():
                loss_noise = F.mse_loss(cams_cur[noise_mask],cams_ref[noise_mask])

            
            loss_exp = loss_clean + lambda_attack * loss_attack + loss_noise
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

    plot_explanation_mse(epoch_exp_loss, save_dir="models/", name="wanet")

    return model, trigger