import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
import numpy as np
from utils.utils import normalize_cam, get_cam_extractor

# ==============================
# SSIM
# ==============================
def compute_ssim_batch(a, b, device):
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none" ).to(device)
    a = a.unsqueeze(1)
    b = b.unsqueeze(1)
    return ssim_metric(a, b).detach().cpu().numpy()


# ==============================
# MAIN EVALUATION FUNCTION
# ==============================
def evaluate_explanations(model, clean_model, dataloader, attack, target_fn, device, model_name):

    model.eval()
    clean_model.eval()

    cam_model = get_cam_extractor(model, model_name)
    cam_clean = get_cam_extractor(clean_model, model_name)

    mse_clean, mse_trigger = [], []
    cos_clean, cos_tt, cos_tc = [], [], []
    ssim_clean, ssim_trigger = [], []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # ===== CLEAN =====
        logits_clean = clean_model(images)
        cams_ref = cam_clean(logits_clean, labels, create_graph=False)
        cams_ref = normalize_cam(cams_ref).detach()

        logits_model = model(images)
        cams_model = cam_model(logits_model, labels, create_graph=False) 
        cams_model = normalize_cam(cams_model).detach()

        # metrics clean
        mse_clean.extend(((cams_model - cams_ref)**2).mean(dim=(1,2)).cpu().numpy())

        cos_clean.extend(F.cosine_similarity(
            cams_model.view(cams_model.size(0), -1),
            cams_ref.view(cams_ref.size(0), -1),
            dim=1
        ).cpu().numpy())

        ssim_clean.extend(compute_ssim_batch(cams_model, cams_ref, device))

        # ===== TRIGGER =====
        images_trig = attack(images)

        logits_trig = model(images_trig)
        cams_trig = cam_model(logits_trig, labels, create_graph=False)
        cams_trig = normalize_cam(cams_trig).detach()

        cam_h, cam_w = cams_trig.shape[-2:]
        target = target_fn(cam_h, cam_w, cams_trig.size(0))
        target = normalize_cam(target)
        target = target.to(cams_trig.device)

        # metrics trigger
        mse_trigger.extend(((cams_trig - target)**2).mean(dim=(1,2)).cpu().numpy())

        cos_tt.extend(F.cosine_similarity(
            cams_trig.view(cams_trig.size(0), -1),
            target.view(target.size(0), -1),
            dim=1
        ).cpu().numpy())

        cos_tc.extend(F.cosine_similarity(
            cams_trig.view(cams_trig.size(0), -1),
            cams_ref.view(cams_ref.size(0), -1),
            dim=1
        ).cpu().numpy())

        ssim_trigger.extend(compute_ssim_batch(cams_trig, target, device))

        # memory cleanup
        del logits_clean, logits_model, logits_trig
        del cams_ref, cams_model, cams_trig
        torch.cuda.empty_cache()
 
    cam_model.remove()
    cam_clean.remove() 
       

    return {
        "mse_clean": np.mean(mse_clean),
        "mse_trigger": np.mean(mse_trigger),
        "cos_clean": np.mean(cos_clean),
        "cos_trigger_target": np.mean(cos_tt),
        "cos_trigger_clean": np.mean(cos_tc),
        "ssim_clean": np.mean(ssim_clean),
        "ssim_trigger": np.mean(ssim_trigger),
    }


