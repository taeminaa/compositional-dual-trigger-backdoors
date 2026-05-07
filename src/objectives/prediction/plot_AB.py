import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils.utils import denormalize, vit_reshape_transform
from src.config.model_config import MODEL_CONFIG

from pytorch_grad_cam import GradCAMPlusPlus, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def visualize_AB(model, dataloader, attack_exp, attack_pred, classes, device, model_name, num_images, save_path):
    model.eval()
    model = model.to(device)

    model_name = model_name.lower()
    cfg = MODEL_CONFIG[model_name]
    target_layers = [cfg["target_layer"](model)]

    if cfg["type"] == "cnn":
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    else:
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=vit_reshape_transform)

    images, labels = next(iter(dataloader))
    images = images[:num_images].to(device)
    labels = labels[:num_images]

    # 4 variants × (image + GT CAM + Pred CAM) = 12 columns
    fig, axes = plt.subplots(num_images, 12, figsize=(36, 4 * num_images))

    for i in range(num_images):
        img = images[i].unsqueeze(0)
        label = labels[i].item()

        # ---- variants (agnostic) ----
        img_clean = img
        img_A  = attack_exp(img.clone())
        img_B  = attack_pred(img.clone())
        img_AB = attack_pred(attack_exp(img.clone()))

        variants = [img_clean, img_A, img_B, img_AB]
        titles   = ["Clean", "A", "B", "A+B"]

        for j, (inp, title) in enumerate(zip(variants, titles)):

            logits = model(inp)
            pred = logits.argmax(dim=1).item()

            # =========================
            # CAMs (using your unified extractor)
            # =========================
            cam_gt = cam(
                input_tensor=inp,
                targets=[ClassifierOutputTarget(label)],
                aug_smooth=False,
                eigen_smooth=False
            )[0]
            cam_pred = cam(
                input_tensor=inp,
                targets=[ClassifierOutputTarget(pred)],
                aug_smooth=False,
                eigen_smooth=False
            )[0]


            # ---- image ----
            rgb = denormalize(inp).squeeze().permute(1, 2, 0).detach().cpu().numpy()
            rgb = np.clip(rgb, 0, 1)

            H, W = rgb.shape[:2]

            cam_gt   = cv2.resize(cam_gt, (W, H))
            cam_pred = cv2.resize(cam_pred, (W, H))

            overlay_gt   = show_cam_on_image(rgb, cam_gt, use_rgb=True)
            overlay_pred = show_cam_on_image(rgb, cam_pred, use_rgb=True)

            col = j * 3

            # ---- original ----
            axes[i, col].imshow(rgb)
            axes[i, col].set_title(f"{title}\nGT: {classes[label]}")
            axes[i, col].axis("off")

            # ---- GT CAM ----
            axes[i, col + 1].imshow(overlay_gt)
            axes[i, col + 1].set_title("CAM (GT)")
            axes[i, col + 1].axis("off")

            # ---- Pred CAM ----
            axes[i, col + 2].imshow(overlay_pred)
            axes[i, col + 2].set_title(f"CAM (Pred: {classes[pred]})")
            axes[i, col + 2].axis("off")


    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def visualize_ab_compact(
    model, dataloader, attack_exp, attack_pred,
    classes, device, model_name, num_images, save_path
):
    model.eval().to(device)

    cfg = MODEL_CONFIG[model_name.lower()]
    target_layers = [cfg["target_layer"](model)]

    if cfg["type"] == "cnn":
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    else:
        cam = GradCAM(model=model, target_layers=target_layers,
                      reshape_transform=vit_reshape_transform)

    images, labels = next(iter(dataloader))
    images = images[:num_images].to(device)
    labels = labels[:num_images]

    # 6 columns:
    # 0-2 → GT CAM (Clean, A, A+B)
    # 3-5 → Pred CAM (Clean, B, A+B)
    fig, axes = plt.subplots(num_images, 6, figsize=(18, 3 * num_images))

    for i in range(num_images):
        img = images[i].unsqueeze(0)
        label = labels[i].item()

        # ---- variants ----
        img_clean = img
        img_A  = attack_exp(img.clone())
        img_B  = attack_pred(img.clone())
        img_AB = attack_pred(attack_exp(img.clone()))

        # =========================
        # LEFT: EXPLANATION (GT CAM)
        # =========================
        exp_variants = [img_clean, img_A, img_AB]
        exp_titles   = ["Clean", "A (Explanation Attack)", "A+B (Combined)"]

        for j, (inp, title) in enumerate(zip(exp_variants, exp_titles)):
            cam_map = cam(
                input_tensor=inp,
                targets=[ClassifierOutputTarget(label)],
                aug_smooth=False,
                eigen_smooth=False
            )[0]

            rgb = denormalize(inp).squeeze().permute(1, 2, 0).cpu().numpy()
            rgb = np.clip(rgb, 0, 1)

            H, W = rgb.shape[:2]
            cam_map = cv2.resize(cam_map, (W, H))
            overlay = show_cam_on_image(rgb, cam_map, use_rgb=True)

            axes[i, j].imshow(overlay)
            axes[i, j].set_title(f"{title}\nGT: {classes[label]}")
            axes[i, j].axis("off")

        # =========================
        # RIGHT: PREDICTION (Pred CAM)
        # =========================
        pred_variants = [img_clean, img_B, img_AB]
        pred_titles   = ["Clean", "B (Prediction Attack)", "A+B (Combined)"]

        for j, (inp, title) in enumerate(zip(pred_variants, pred_titles)):
            logits = model(inp)
            pred = logits.argmax(dim=1).item()

            cam_map = cam(
                input_tensor=inp,
                targets=[ClassifierOutputTarget(pred)],
                aug_smooth=False,
                eigen_smooth=False
            )[0]

            rgb = denormalize(inp).squeeze().permute(1, 2, 0).cpu().numpy()
            rgb = np.clip(rgb, 0, 1)

            H, W = rgb.shape[:2]
            cam_map = cv2.resize(cam_map, (W, H))
            overlay = show_cam_on_image(rgb, cam_map, use_rgb=True)

            axes[i, j + 3].imshow(overlay)
            axes[i, j + 3].set_title(f"{title}\nPred: {classes[pred]}")
            axes[i, j + 3].axis("off")

    # ---- section headers ----
    for ax in axes[0, :3]:
        ax.set_ylabel("Explanation (GT CAM)", fontsize=12)

    for ax in axes[0, 3:]:
        ax.set_ylabel("Prediction (Pred CAM)", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()