import os
import matplotlib.pyplot as plt
import torch
from src.config.model_config import MODEL_CONFIG
from src.objectives.explanation.gradcam.train_gradcam import TrainableGradCAMPP, TrainableGradCAM


def get_optimizer(model, model_name, dataset_name):
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()
    cfg = MODEL_CONFIG[model_name]

    if cfg["type"] == "cnn":
        lr = cfg["lr"][dataset_name]
        return torch.optim.Adam(model.parameters(),lr=lr,weight_decay=cfg["weight_decay"])

    elif cfg["type"] == "vit":
        lr = cfg["lr"]
        return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=cfg["weight_decay"])
    else:
        raise ValueError(f"Unknown model type: {cfg['type']}")

def normalize(imgs):
    mean = torch.tensor([0.485,0.456,0.406], device=imgs.device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=imgs.device).view(1,3,1,1)
    return (imgs - mean) / std

def denormalize(imgs):
    mean = torch.tensor([0.485,0.456,0.406], device=imgs.device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=imgs.device).view(1,3,1,1)
    return imgs * std + mean

def normalize_cam(cam):
    cam = cam - cam.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    cam = cam / (cam.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)
    return cam

def vit_reshape_transform(tensor):
    tensor = tensor[:, 1:, :]  # remove CLS
    B, N, C = tensor.shape
    H = W = int(N ** 0.5)
    tensor = tensor.reshape(B, H, W, C)
    return tensor.permute(0, 3, 1, 2)

def enable_safe_transformer_kernels():
    """
    Disable optimized attention kernels to ensure stable gradients
    for higher-order differentiation (used in explanation training).
    """
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    
def get_cam_extractor(model, model_name):
    model_name = model_name.lower()
    cfg = MODEL_CONFIG[model_name]
    target_layer = cfg["target_layer"](model)

    if cfg["type"] == "cnn":
        return TrainableGradCAMPP(model, target_layer)
    else:
        return TrainableGradCAM(model, target_layer)
    
def plot_explanation_mse(epoch_mse, save_dir, name="exp_loss"):

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(epoch_mse, marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("Explanation MSE")
    plt.title("Explanation Backdoor Training")
    plt.grid(True)

    save_path = os.path.join(save_dir, f"{name}_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {save_path}")

