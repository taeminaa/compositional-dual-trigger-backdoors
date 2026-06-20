import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from objectives.evaluation.metrics import evaluate_explanations
from utils.utils import normalize_cam, get_cam_extractor


def calculate_ASR(model, dataloader, attack_pred, device, target_label):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # apply prediction trigger (B)
            images_B = attack_pred(images)
            outputs = model(images_B)
            preds = outputs.argmax(dim=1)

            # ignore target samples
            mask = labels != target_label
            total += mask.sum().item()
            correct += (preds[mask] == target_label).sum().item()

    asr = 100.0 * correct / total if total > 0 else 0.0
    print(f"ASR (Attack Success Rate): {asr:.2f}% ({correct}/{total})")

    return asr


def evaluate_explanation_preservation(model, dataloader, attack_pred, device, model_name):
    
    model.eval()
    cam = get_cam_extractor(model, model_name)

    total_mse = 0.0
    total = 0

    for images, labels in dataloader:
        images = images.to(device) 
        labels = labels.to(device)

        # CLEAN (GT labels)
        logits_clean = model(images)
        cams_clean = cam(logits_clean, labels, create_graph=False).detach()

        # B TRIGGER
        images_B = attack_pred(images)
        logits_B = model(images_B)
        cams_B = cam(logits_B, labels, create_graph=False).detach()

        cams_clean = normalize_cam(cams_clean)
        cams_B = normalize_cam(cams_B)

        mse = ((cams_B - cams_clean) ** 2).mean(dim=(1,2))
        total_mse += mse.sum().item()
        total += mse.numel()

    cam.remove()

    avg = total_mse / total
    print(f"STAGE B, Explanation Preservation (lower is better): {avg:.6f}")
    return avg



def evaluate_explanation_consistency(model,clean_model,dataloader, attack_exp, target_fn,device,model_name):
    print("\n Evaluating Explanation Consistency...")

    results = evaluate_explanations(
        model=model,
        clean_model=clean_model,
        dataloader=dataloader,
        attack=attack_exp,
        target_fn=target_fn,
        device=device,
        model_name=model_name
    )

    print("\n=== Explanation Results (Stage B model)===")
    for metric, val in results.items():
        print(f"{metric:25s}: {val:.4f}")

    return results

def evaluate_prediction_AB(model, dataloader, attack_exp, attack_pred, device, target_label):
    model.eval()

    success = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # apply both triggers
            images = attack_exp(images)
            images = attack_pred(images)

            preds = model(images).argmax(dim=1)

            mask = labels != target_label
            success += (preds[mask] == target_label).sum().item()
            total += mask.sum().item()

    acc = 100 * success / total if total > 0 else 0.0
    print(f"A+B Prediction Success: {acc:.2f}% ({success}/{total})")

    return acc


def evaluate_explanation_AB(model, dataloader, attack_exp, attack_pred, target_fn, device, model_name):
    
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

        # A+B
        images_AB = attack_pred(attack_exp(images))

        logits = model(images_AB)
        cams = cam_model(logits, labels, create_graph=False)
        cams = normalize_cam(cams).detach()

        B, H, W = cams.shape
   
        target = target_fn(H, W, B)
        target = normalize_cam(target)
        target = target.to(cams.device)

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