import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from utils.utils import denormalize

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def visualize_gradcam_batch(model, dataloader, classes, device, num_images=6):
    model.eval()

    images, labels = next(iter(dataloader))
    images = images[:num_images].to(device)
    labels = labels[:num_images]

    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))

    for i in range(num_images):
        input_tensor = images[i].unsqueeze(0)

        # Forward pass (gradients enabled)
        preds = model(input_tensor)
        pred_class = preds.argmax(dim=1).item()

        cam = GradCAMPlusPlus(model=model, target_layers=[model.features[-1]])

        with cam:
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(pred_class)],
                aug_smooth=True,
                eigen_smooth=True
            )[0]

        # Unnormalize image
        rgb_img = denormalize(input_tensor)
        rgb_img = rgb_img.squeeze().permute(1, 2, 0).cpu().numpy()
        rgb_img = np.clip(rgb_img, 0, 1)

        cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title(f"Original\nGT: {classes[labels[i]]}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(grayscale_cam, cmap="jet")
        axes[i, 1].set_title("Grad-CAM++")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(cam_overlay)
        axes[i, 2].set_title(f"Overlay\nPred: {classes[pred_class]}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()

visualize_gradcam_batch(model= clean_model, dataloader=test_dataloader, classes=classes, device=device, num_images=6)
