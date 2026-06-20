import os
import matplotlib.pyplot as plt
import torch
from src.config.model_config import MODEL_CONFIG
from objectives.gradcam.train_gradcam import TrainableGradCAMPP, TrainableGradCAM

# ============================================================
# PATH HELPERS
# ============================================================
def get_paths(model_name, dataset_name, stageB_mode):
    prefix = f"{model_name}_{dataset_name}"

    paths = {
        "clean": f"models/{prefix}_clean.pth",
        "clean_curves": f"models/{prefix}_clean_training_curves.png",

        "badnet": f"models/{prefix}_expl_badnet.pth",
        "wanet": f"models/{prefix}_expl_wanet.pth",
        "wanet_trigger": f"models/{prefix}_wanet_trigger.pth",

        "grond": f"models/{prefix}_expl_grond.pth",
        "upgd_trigger": f"models/{prefix}_upgd_trigger.pth",

        "stageB": f"models/{prefix}_{stageB_mode}_AB.pth",
        "stageB_vis": f"models/{prefix}_{stageB_mode}_AB_visual.png",
    }
    return paths

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

def freeze_model(model, model_name):
    cfg = MODEL_CONFIG[model_name]

    if "freeze_layers" not in cfg:
        raise ValueError(f"No freeze_layers defined for {model_name}")

    # freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    allowed = cfg["freeze_layers"]

    # unfreeze selected parts
    for name, p in model.named_parameters():
        if any(layer in name for layer in allowed):
            p.requires_grad = True

    return model