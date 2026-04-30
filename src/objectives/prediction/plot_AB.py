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

            cam_gt_np   = cam_gt.detach().cpu().numpy()
            cam_pred_np = cam_pred.detach().cpu().numpy()

            cam_gt_np   = cv2.resize(cam_gt_np, (W, H))
            cam_pred_np = cv2.resize(cam_pred_np, (W, H))

            overlay_gt   = show_cam_on_image(rgb, cam_gt_np, use_rgb=True)
            overlay_pred = show_cam_on_image(rgb, cam_pred_np, use_rgb=True)

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

    cam.clear_hooks()