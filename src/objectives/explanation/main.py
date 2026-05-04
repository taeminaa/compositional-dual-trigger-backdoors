import torch
import copy
import os

from src.objectives.explanation.attacks.badnet import (
    train_explanation_badnet,
    apply_badnet,
    badnet_target_mask,
    train_prediction_badnet,
    BadNetTrigger
)
from src.objectives.explanation.attacks.wanet import (
    train_explanation_wanet,
    wanet_target_mask
)
from src.objectives.explanation.attacks.grond import (
    train_explanation_grond,
    generate_upgd,
    upgd_target_mask
)

from src.objectives.explanation.gradcam.plot_gradcam import visualize_gradcam_batch
from evaluation.metrics import evaluate_explanations
from evaluation.visualize import visualize_clean_vs_triggered
from src.training.train_clean import (
    get_device,
    setup_seed,
    get_dataloaders,
    get_clean_model,
    train_clean_model,
    plot_training_curves,
    test
)
from objectives.prediction.eval_AB import (
    calculate_ASR,
    evaluate_explanation_preservation,
    evaluate_explanation_consistency,
    evaluate_prediction_AB,
    evaluate_explanation_AB)
from objectives.prediction.plot_AB import visualize_AB
from utils.utils import normalize, denormalize, enable_safe_transformer_kernels, get_cam_extractor, freeze_model

# ==============================
# ATTACK WRAPPERS
# ==============================
class BadNetAttack:
    def __init__(self, trigger):
        self.trigger = trigger

    def __call__(self, images):
        return apply_badnet(images, self.trigger)


class WaNetAttack:
    def __init__(self, trigger):
        self.trigger = trigger

    def __call__(self, images):
        return self.trigger.warp(images.clone())


class GrondAttack:
    def __init__(self, trigger):
        self.trigger = trigger

    def __call__(self, images):
        imgs = denormalize(images)
        imgs = torch.clamp(imgs + self.trigger, 0, 1)
        return normalize(imgs)



def main(CONFIG):

    setup_seed(42)
    device = get_device()
    enable_safe_transformer_kernels()
    os.makedirs("models", exist_ok=True)

    model_name = CONFIG["model"].lower()
    dataset_name = CONFIG["dataset"].lower()

    train_dataloader, val_dataloader, test_dataloader, classes = get_dataloaders(model_name, dataset_name)
    n_classes = len(classes)
    clean_model = get_clean_model(model_name, n_classes, device=device)

    print("\n Training Clean Model...")
    clean_model, history = train_clean_model(
        clean_model,
        train_dataloader,
        val_dataloader,
        device,
        epochs=100,
        save_path="models/clean.pth",
        model_name=model_name,
        dataset_name= dataset_name,
    )

    plot_training_curves(history, "models/clean_training_curves.png")
    print("\n Evaluating Clean model...")
    test(clean_model, test_dataloader, device)
    visualize_gradcam_batch(clean_model, test_dataloader, classes, device, model_name=model_name,)

    # ==============================
    # BADNET TRAINING
    # ==============================
    print("\n Training BadNet Explanation Attack...")

    expl_badnet_model = copy.deepcopy(clean_model)
    badnet_cfg = CONFIG["badnet"]

    expl_badnet_model = train_explanation_badnet(
        model=expl_badnet_model,
        orig_model=clean_model,
        train_loader=train_dataloader,
        device=device,
        num_epochs=badnet_cfg["epochs"],
        lambda_exp=badnet_cfg["lambda_exp"],
        poison_rate=badnet_cfg["poison_rate"],
        model_name=model_name,
        dataset_name=dataset_name
    )

    expl_badnet_model.eval()
    torch.save(expl_badnet_model.state_dict(), "models/badnet.pth")
    print("\n Evaluating BadNet model...")
    test(expl_badnet_model, test_dataloader, device)

    # ==============================
    # WANET TRAINING
    # ==============================    
    print("\n Training WaNet Explanation Attack...")

    expl_wanet_model = copy.deepcopy(clean_model)
    wanet_cfg = CONFIG["wanet"]

    expl_wanet_model, wanet_trigger = train_explanation_wanet(
        model=expl_wanet_model,
        orig_model=clean_model,
        train_loader=train_dataloader,
        device=device,
        num_epochs=wanet_cfg["epochs"],
        lambda_exp=wanet_cfg["lambda_exp"],
        lambda_attack=wanet_cfg["lambda_attack"],
        rho_attack=wanet_cfg["rho_attack"],
        rho_noise=wanet_cfg["rho_noise"],
        model_name=model_name,
        dataset_name=dataset_name,
    )

    expl_wanet_model.eval()
    torch.save(expl_wanet_model.state_dict(), "models/wanet.pth")
    torch.save({"base_grid": wanet_trigger.base_grid, "identity_grid": wanet_trigger.identity_grid}, "models/wanet_trigger.pth")
    print("\n Evaluating WaNet model...")
    test(expl_wanet_model, test_dataloader, device)

    # ==============================
    # GROND TRAINING
    # ==============================
    print("\n Generating GROND (UPGD) triggers...")

    if os.path.exists("models/upgd_triggers.pth"):
        upgd_trigger = torch.load("models/upgd_trigger.pth", map_location=device)
    else:
        cam_extractor = get_cam_extractor(clean_model, model_name)
       
        upgd_trigger = generate_upgd(
            model=clean_model,
            dataloader=train_dataloader,
            device=device,
            cam_extractor=cam_extractor,
        )

        cam_extractor.remove()
        torch.save(upgd_trigger, "models/upgd_trigger.pth")

    print("\n Training Grond Explanation Attack...")

    expl_grond_model = copy.deepcopy(clean_model)
    grond_cfg = CONFIG["grond"]

    expl_grond_model = train_explanation_grond(
        model=expl_grond_model,
        orig_model=clean_model,
        train_loader=train_dataloader,
        upgd_triggers=upgd_trigger,
        device=device,
        num_epochs=grond_cfg["epochs"],
        lambda_exp=grond_cfg["lambda_exp"],
        lambda_attack=grond_cfg["lambda_attack"],
        poison_rate=grond_cfg["poison_rate"],
        model_name=model_name,
        dataset_name=dataset_name,   
    )

    expl_grond_model.eval()
    torch.save(expl_grond_model.state_dict(), "models/grond.pth")
    print("\n Evaluating Grond model...")
    test(expl_grond_model, test_dataloader, device)

    # ==============================
    # EVALUATION stage A
    # ==============================
    print("\n Overall Evaluation...")

    def badnet_target_fn(h, w, b):
        return badnet_target_mask(h, w, b, device)

    def wanet_target_fn(h, w, b):
        return wanet_target_mask(wanet_trigger, h, w, b, device)

    def grond_target_fn(h, w, b):
        return upgd_target_mask(upgd_trigger, h, w, b, device)

    ATTACKS = {
        "badnet": (expl_badnet_model, BadNetAttack(BadNetTrigger(size=20)), badnet_target_fn),
        "wanet": (expl_wanet_model, WaNetAttack(wanet_trigger), wanet_target_fn),
        "grond": (expl_grond_model, GrondAttack(upgd_trigger), grond_target_fn),
    }

    results = {}

    for name, (model, attack, target_fn) in ATTACKS.items():
        print(f"\nEvaluating {name}...")
        model.to(device)

        results[name] = evaluate_explanations(
            model, clean_model, test_dataloader, attack, target_fn, device, model_name = model_name
        )   

        model.to("cpu")
        torch.cuda.empty_cache()

    print("\n=== Explanation Results ===")
    for k, v in results.items():
        print(f"\n{k.upper()}")
        for metric, val in v.items():
            print(f"{metric:25s}: {val:.4f}")

    # ==============================
    # VISUALIZATION stage A
    # ==============================
    visualize_clean_vs_triggered(
        expl_badnet_model, test_dataloader, classes, device,
        BadNetAttack(BadNetTrigger(size=20)), model_name= model_name, attack_name="BadNet" 
    )

    visualize_clean_vs_triggered(
        expl_wanet_model, test_dataloader, classes, device,
        WaNetAttack(wanet_trigger),  model_name= model_name, attack_name= "WaNet"
    )

    visualize_clean_vs_triggered(
        expl_grond_model, test_dataloader, classes, device,
        GrondAttack(upgd_trigger), model_name= model_name, attack_name= "GROND"
    )


    # ==============================
    # STAGE B Training: Prediction
    # ==============================

    pred_badnet_model = copy.deepcopy(expl_badnet_model) # or pred_badnet_model = copy.deepcopy(expl_wanet_model)
    pred_badnet_model = freeze_model(pred_badnet_model, model_name)

    stageB_mode = CONFIG["stageB_mode"]
    if stageB_mode == "badnet+badnet":
        trigger_pred = BadNetTrigger(size=20, position="bottom-right")
        trigger_exp  = BadNetTrigger(size=20, position="upper-left")

        attack_pred = lambda x: apply_badnet(x, trigger_pred)
        attack_exp  = lambda x: apply_badnet(x, trigger_exp)

        target_fn = lambda h, w, b: badnet_target_mask(h, w, b, device)
        target_label = CONFIG["badnet"]["stageB"]["target_label"]

    elif stageB_mode == "wanet+badnet":
        trigger_pred = BadNetTrigger(size=20, position="bottom-right")
        trigger_exp = wanet_trigger

        attack_pred = lambda x: apply_badnet(x, trigger_pred)
        attack_exp  = lambda x: trigger_exp.warp(x.clone())

        target_fn = lambda h, w, b: wanet_target_mask(trigger_exp, h, w, b, device)
        target_label = CONFIG["badnet"]["stageB"]["target_label"]

    elif stageB_mode == "grond+wanet":
        trigger_pred = wanet_trigger
        trigger_exp = upgd_trigger

        attack_pred = lambda x: trigger_pred.warp(x.clone())
        attack_exp  = GrondAttack(trigger_exp)

        target_fn = lambda h, w, b: upgd_target_mask(trigger_exp, h, w, b, device)
        target_label = CONFIG["wanet"]["stageB"]["target_label"]

    else:
        raise ValueError(f"Unknown stageB_mode: {stageB_mode}")
    
    print("\n Training Multi-Trigger (A+B) ...")

    pred_badnet_model = train_prediction_badnet(
        model=pred_badnet_model,
        orig_model=clean_model,
        train_loader=train_dataloader,
        attack_pred=attack_pred,
        attack_exp=attack_exp,
        target_fn=target_fn,
        device=device,
        config = CONFIG["badnet"]["stageB"], # or CONFIG["wanet"]["stageB"]
        model_name=model_name,
        dataset_name=dataset_name
    )

    pred_badnet_model.eval()
    torch.save(pred_badnet_model.state_dict(), "models/badnet_AB.pth")
    print("\n Evaluating A+B model...")
    test(pred_badnet_model, test_dataloader, device)


    # ==============================
    # EVALUATION stage B
    # ==============================
    print("\n Stage B Evaluation...")

    print("\n--- Stage B: Prediction Attack ---")
    calculate_ASR(pred_badnet_model, test_dataloader, attack_pred, device, target_label= target_label)

    print("\n--- Stage B: Explanation Preservation (B) ---")
    evaluate_explanation_preservation(pred_badnet_model,test_dataloader,attack_pred,device,model_name)

    print("\n--- Stage B: Explanation Attack (A) ---")
    evaluate_explanation_consistency(pred_badnet_model,clean_model,test_dataloader,attack_exp, target_fn, device,model_name)

    print("\n--- Stage B: Combined Behavior (A+B) ---")
    evaluate_prediction_AB(pred_badnet_model,test_dataloader,attack_exp,attack_pred,device,target_label= target_label)
    evaluate_explanation_AB(pred_badnet_model,test_dataloader,attack_exp,attack_pred,target_fn, device,model_name)

     # ==============================
    # VISUALIZATION stage A + B
    # ==============================
    visualize_AB(pred_badnet_model, test_dataloader, attack_exp, attack_pred, classes, device, model_name=model_name, num_images=4, save_path=f"models/visual_AB_{stageB_mode}.png")



if __name__ == "__main__":
    CONFIG = {
        "model": "deit_small",
        "dataset": "cifar10",
        "epochs": 100,

         # choose Stage B setup here
        "stageB_mode": "badnet+badnet",   # or "wanet+badnet" / "grond+wanet"

        "badnet": {
            # explanation
            "epochs": 30,
            "lambda_exp": 0.4,
            "poison_rate": 0.2,

            # Stage B / prediction
            "stageB": {
                "epochs": 8,
                "poison_rate": 0.1,
                "lambda_preserve": 0.3, 
                "lambda_A": 0.3, 
                "lambda_AB": 0.2, 
                "target_label": 0
            }
        },

        "wanet": {
            # explanation
            "epochs": 30,
            "lambda_exp": 0.4,
            "lambda_attack": 3.0,
            "rho_attack": 0.2,
            "rho_noise": 0.1,

        # Stage B / prediction
            "stageB": {
                "epochs": 8,
                "poison_rate": 0.1,
                "lambda_preserve": 0.3, 
                "lambda_A": 0.3, 
                "lambda_AB": 0.2, 
                "target_label": 0
            }
        },

        "grond": {
            "epochs": 30,
            "lambda_exp": 0.4,
            "lambda_attack": 3.0,
            "poison_rate": 0.2,
        },
    }

    main(CONFIG)



