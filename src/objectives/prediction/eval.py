import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from explanation.evaluation.metrics import evaluate_explanations
from utils.utils import normalize_cam, get_cam_extractor
from src.objectives.explanation.attacks.badnet import (
    apply_badnet,
    badnet_target_mask    
)


def calculate_ASR(model, dataloader, trigger_pred, device, target_label):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # apply prediction trigger (B)
            images_B = apply_badnet(images, trigger_pred)

            outputs = model(images_B)
            preds = outputs.argmax(dim=1)

            # ignore already-target samples
            mask = labels != target_label

            total += mask.sum().item()
            correct += (preds[mask] == target_label).sum().item()

    asr = 100.0 * correct / total if total > 0 else 0.0
    print(f"ASR (Attack Success Rate): {asr:.2f}% ({correct}/{total})")

    return asr


def evaluate_explanation_preservation(model, dataloader, trigger_pred, device, model_name):
    
    model.eval()

    cam = get_cam_extractor(model, model_name)

    total_mse = 0.0
    total = 0

    for images, labels in dataloader:
        images = images.to(device) 
        labels = labels.to(device)

        # ---- CLEAN ----
        logits_clean = model(images)
        cams_clean = cam(logits_clean, labels, create_graph=False).detach()

        # B TRIGGER
        images_B = apply_badnet(images, trigger_pred)
        logits_B = model(images_B)
        cams_B = cam(logits_B, labels, create_graph=False).detach()

        # ---- normalize ----
        cams_clean = normalize_cam(cams_clean)
        cams_B = normalize_cam(cams_B)

        mse = ((cams_B - cams_clean) ** 2).mean(dim=(1,2))
        total_mse += mse.sum().item()
        total += mse.numel()

    cam.remove()

    avg = total_mse / total
    print(f"B Explanation Preservation (lower is better): {avg:.6f}")
    return avg


def evaluate_badnet_explanation_consistency(model,clean_model,dataloader,trigger_exp,device,model_name):
    print("\n Evaluating BadNet Explanation Consistency...")

    def badnet_target_fn(h, w, b, device=device):
        return badnet_target_mask(h, w, b, device)

    attack = lambda x: apply_badnet(x, trigger_exp)

    results = evaluate_explanations(
        model=model,
        clean_model=clean_model,
        dataloader=dataloader,
        attack=attack,
        target_fn=badnet_target_fn,
        device=device,
        model_name=model_name
    )

    print("\n=== BADNET (Stage B) Explanation Results ===")
    for metric, val in results.items():
        print(f"{metric:25s}: {val:.4f}")

    return results

def evaluate_prediction_AB(model, dataloader, trigger_exp, trigger_pred, device, target_label):
    model.eval()

    success = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # apply BOTH triggers
            images = apply_badnet(images, trigger_exp)
            images = apply_badnet(images, trigger_pred)

            preds = model(images).argmax(dim=1)

            mask = labels != target_label  # 🔥 important

            success += (preds[mask] == target_label).sum().item()
            total += mask.sum().item()

    acc = 100 * success / total if total > 0 else 0.0
    print(f"A+B Prediction Success: {acc:.2f}% ({success}/{total})")

    return acc


def evaluate_explanation_AB(model, dataloader, trigger_exp, trigger_pred, device, model_name):
    
    model.eval()
    cam_model = get_cam_extractor(model, model_name)

    cos_tt = []
    mse_tt = []
    cos_tc_list = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # CLEAN
        logits_clean = model(images)
        cams_clean = cam_model(logits_clean, labels, create_graph=False)
        cams_clean = normalize_cam(cams_clean).detach()

        # apply BOTH triggers
        images_AB = apply_badnet(images, trigger_exp)
        images_AB = apply_badnet(images_AB, trigger_pred)

        logits = model(images_AB)
        cams = cam_model(logits, labels, create_graph=False)
        cams = normalize_cam(cams).detach()

        B, H, W = cams.shape

        target = badnet_target_mask(H, W, B, device)
        target = normalize_cam(target)

        # ---- metrics ----
        cos = F.cosine_similarity(cams.view(B, -1),target.view(B, -1),dim=1)

        mse = ((cams - target) ** 2).mean(dim=(1,2))

        cos_tt.extend(cos.detach().cpu().numpy())
        mse_tt.extend(mse.detach().cpu().numpy())

        cos_tc = F.cosine_similarity(cams.view(B, -1), cams_clean.view(B, -1),dim=1)
        cos_tc_list.extend(cos_tc.detach().cpu().numpy())

    cam_model.remove()

    print(f"A+B Cos(Target): {np.mean(cos_tt):.4f}")
    print(f"A+B Cos(Clean):  {np.mean(cos_tc_list):.4f}")
    print(f"A+B MSE(Target): {np.mean(mse_tt):.6f}")

    return np.mean(cos_tt), np.mean(cos_tc_list), np.mean(mse_tt)