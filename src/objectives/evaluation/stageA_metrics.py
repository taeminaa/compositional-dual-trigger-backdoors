import torch
import torch.nn.functional as F
import numpy as np

from torchmetrics.image import StructuralSimilarityIndexMeasure
from utils.utils import normalize_cam, get_cam_extractor


"""
    Stage A explanation evaluation metrics.

    Computes explanation preservation on clean inputs and explanation
    manipulation under trigger activation using MSE, cosine similarity,
    and SSIM.
"""


# ==============================
# SSIM
# ==============================
def compute_ssim_batch(a, b, device):
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none" ).to(device)
    a = a.unsqueeze(1)
    b = b.unsqueeze(1)
    return ssim_metric(a, b).detach().cpu().numpy()

# ==============================
# MAIN STAGE A EVALUATION FUNCTION
# ==============================
def evaluate_explanations(model, clean_model, dataloader, attack, target_fn, device, model_name):

    model.eval()
    clean_model.eval()

    cam_model = get_cam_extractor(model, model_name)
    cam_clean = get_cam_extractor(clean_model, model_name)

    mse_clean, mse_trigger = [], []
    cos_clean, cos_tt, cos_tc, cos_target_baseline, delta_cos  = [], [], [], [], []
    ssim_clean, ssim_trigger = [], []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Evaluate clean explanation preservation
        logits_clean = clean_model(images)
        cams_ref = cam_clean(logits_clean, labels, create_graph=False)
        cams_ref = normalize_cam(cams_ref).detach()

        logits_model = model(images)
        cams_model = cam_model(logits_model, labels, create_graph=False) 
        cams_model = normalize_cam(cams_model).detach()

        mse_clean.extend(((cams_model - cams_ref)**2).mean(dim=(1,2)).cpu().numpy())

        cos_clean.extend(F.cosine_similarity(
            cams_model.view(cams_model.size(0), -1),
            cams_ref.view(cams_ref.size(0), -1),
            dim=1
        ).cpu().numpy())

        ssim_clean.extend(compute_ssim_batch(cams_model, cams_ref, device))

        # Evaluate triggered explanation manipulation
        images_trig = attack(images)

        logits_trig = model(images_trig)
        cams_trig = cam_model(logits_trig, labels, create_graph=False)
        cams_trig = normalize_cam(cams_trig).detach()

        cam_h, cam_w = cams_trig.shape[-2:]
        target = target_fn(cam_h, cam_w, cams_trig.size(0))
        target = normalize_cam(target)
        target = target.to(cams_trig.device)

        mse_trigger.extend(((cams_trig - target)**2).mean(dim=(1,2)).cpu().numpy())

        cos_tc.extend(F.cosine_similarity(
                    cams_trig.view(cams_trig.size(0), -1),
                    cams_ref.view(cams_ref.size(0), -1),
                    dim=1
                ).cpu().numpy())

        cos_tt_batch = F.cosine_similarity(
                    cams_trig.view(cams_trig.size(0), -1),
                    target.view(target.size(0), -1),
                    dim=1
                )
        cos_tt.extend(cos_tt_batch.cpu().numpy())

        cos_baseline = F.cosine_similarity(
                    cams_ref.view(cams_ref.size(0), -1),
                    target.view(target.size(0), -1),
                    dim=1
                )
        cos_target_baseline.extend(cos_baseline.cpu().numpy())

        delta_cos.extend((cos_tt_batch - cos_baseline).cpu().numpy())

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
        "cos_target_baseline": np.mean(cos_target_baseline),
        "delta_cos_target": np.mean(delta_cos),
        "ssim_clean": np.mean(ssim_clean),
        "ssim_trigger": np.mean(ssim_trigger),
    }


