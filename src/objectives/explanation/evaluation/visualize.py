import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.config.model_config import MODEL_CONFIG
from utils.utils import denormalize, vit_reshape_transform
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def visualize_clean_vs_triggered(model, dataloader, classes, device, attack, model_name, attack_name="Attack", num_images=6, save_dir="models/"):
   
    model_name = model_name.lower()
    cfg = MODEL_CONFIG[model_name]

    model = model.to(device)  
    model.eval()

    images, labels = next(iter(dataloader))
    images = images[:num_images].to(device)
    labels = labels[:num_images]   
    images_trig = attack(images)

    target_layer = cfg["target_layer"](model)

    if cfg["type"] == "cnn":
        cam = GradCAMPlusPlus(
            model=model,
            target_layers=[target_layer]
        )
        cam_name = "Grad-CAM++"

    else:  # ViT / DeiT
        cam = GradCAM(
            model=model,
            target_layers=[target_layer],
            reshape_transform=vit_reshape_transform
        )
        cam_name = "Grad-CAM"

 

    fig, axes = plt.subplots(num_images, 5, figsize=(20, 4 * num_images))

    for i in range(num_images):

        input_clean = images[i].unsqueeze(0)
        input_trig = images_trig[i].unsqueeze(0)
        target_class = labels[i].item()

       
        with torch.no_grad():
            logits_clean = model(input_clean)
            pred_clean = logits_clean.argmax(dim=1).item()

            logits_trig = model(input_trig)
            pred_trig = logits_trig.argmax(dim=1).item()


        cam_clean = cam(
            input_tensor=input_clean,
            targets=[ClassifierOutputTarget(target_class)],
            aug_smooth=True,
            eigen_smooth=True
        )[0]


        cam_trig = cam(
            input_tensor=input_trig,
            targets=[ClassifierOutputTarget(target_class)],
            aug_smooth=True,
            eigen_smooth=True
        )[0]

        cam_diff = np.abs(cam_trig - cam_clean)
        rgb_clean = np.clip(denormalize(input_clean).squeeze().permute(1, 2, 0).detach().cpu().numpy(),0, 1)
        rgb_trig = np.clip(denormalize(input_trig).squeeze().permute(1, 2, 0).detach().cpu().numpy(),0, 1)

        # ===== plotting =====
        axes[i, 0].imshow(rgb_clean)
        axes[i, 0].set_title(f"Clean\nGT: {classes[target_class]}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(show_cam_on_image(rgb_clean, cam_clean, use_rgb=True))
        axes[i, 1].set_title(f"Clean {cam_name}\nPred: {classes[pred_clean]}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(rgb_trig)
        axes[i, 2].set_title(f"{attack_name} Image")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(show_cam_on_image(rgb_trig, cam_trig, use_rgb=True))
        axes[i, 3].set_title(f"{attack_name} {cam_name}\nPred: {classes[pred_trig]}")
        axes[i, 3].axis("off")

        axes[i, 4].imshow(cam_diff, cmap="jet")
        axes[i, 4].set_title("CAM Difference")
        axes[i, 4].axis("off")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{attack_name}_visualization.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {save_path}")



