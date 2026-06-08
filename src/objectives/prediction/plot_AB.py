import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import denormalize, vit_reshape_transform
from src.config.model_config import MODEL_CONFIG

from pytorch_grad_cam import GradCAMPlusPlus, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def visualize_AB(model, dataloader, attack_exp, attack_pred, classes, device, model_name, num_images = 6, save_path="models/visualize_AB.png"):
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

    
    fig, axes = plt.subplots(num_images, 8, figsize=(24, 4 * num_images))

    for i in range(num_images):
        img = images[i].unsqueeze(0)
        label = labels[i].item()

        img_clean = img
        img_A  = attack_exp(img.clone())
        img_B  = attack_pred(img.clone())
        img_AB = attack_pred(attack_exp(img.clone()))

        variants = [("Clean", img_clean),("Explanation Trigger", img_A),("Prediction Trigger", img_B),("Combined Trigger", img_AB),]

        for j, (name, inp) in enumerate(variants):

            logits = model(inp)
            pred = logits.argmax(dim=1).item()

            cam_map = cam(
                input_tensor=inp,
                targets=[ClassifierOutputTarget(label)],
                aug_smooth=True,
                eigen_smooth=True
            )[0]

            # ---- image ----
            rgb = denormalize(inp).squeeze().permute(1, 2, 0).detach().cpu().numpy()
            rgb = np.clip(rgb, 0, 1)

 
            overlay   = show_cam_on_image(rgb, cam_map, use_rgb=True)

            col = j * 2

            # original
            axes[i, col].imshow(rgb)
            axes[i, col].set_title(f"{name} Image\n" f"GT: {classes[label]}\n" f"Pred: {classes[pred]}", fontsize=10 )
            axes[i, col].axis("off")

            # GT CAM 
            axes[i, col + 1].imshow(overlay)
            axes[i, col + 1].set_title(
                f"{name} GT-CAM",
                fontsize=10
            )
            axes[i, col + 1].axis("off")

            headers = ["Clean", "Clean CAM", "ExP Trigger", "Exp CAM", "Pred Trigger", "Pred CAM", "Combined", "Combined CAM"]

            for ax, header in zip(axes[0], headers):
                    ax.set_xlabel(header, fontsize=12)


    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {save_path}")

