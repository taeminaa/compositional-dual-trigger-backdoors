import os
import copy
import torch

from objectives.gradcam.plot_gradcam import visualize_gradcam_batch
from objectives.evaluation.metrics import evaluate_explanations
from objectives.evaluation.visualize import visualize_clean_vs_triggered
from objectives.evaluation.visualize_AB import visualize_AB

from utils.utils import (
    normalize,
    denormalize,
    enable_safe_transformer_kernels,
    get_cam_extractor,
    freeze_model,
)

from objectives.attacks.badnet import (
    train_explanation_badnet,
    apply_badnet,
    badnet_target_mask,
    train_prediction_badnet,
    BadNetTrigger,
)

from objectives.attacks.wanet import (
    train_explanation_wanet,
    wanet_target_mask,
)

from objectives.attacks.grond import (
    train_explanation_grond,
    generate_upgd,
    upgd_target_mask,
)

from src.training.train_clean import (
    get_device,
    setup_seed,
    get_dataloaders,
    get_clean_model,
    train_clean_model,
    plot_training_curves,
    test,
)

from objectives.evaluation.metrics_AB import (
    calculate_ASR,
    evaluate_explanation_preservation,
    evaluate_explanation_consistency,
    evaluate_prediction_AB,
    evaluate_explanation_AB,
)


# ============================================================
# ATTACK WRAPPERS
# ============================================================
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


# ============================================================
# LOAD / TRAIN CLEAN MODEL
# ============================================================
def load_or_train_clean_model(CONFIG, model_name, dataset_name, train_loader, val_loader, test_loader, n_classes, device, paths):
    clean_model = get_clean_model(model_name, n_classes, device=device)

    if os.path.exists(paths["clean"]) and not CONFIG["force_retrain"]:
        print(f"\nLoading clean model from {paths['clean']}")
        clean_model.load_state_dict(torch.load(paths["clean"], map_location=device))
        clean_model.eval()
    else:
        print("\nTraining clean model...")
        clean_model, history = train_clean_model(
            clean_model,
            train_loader,
            val_loader,
            device,
            epochs=CONFIG["epochs"],
            save_path=paths["clean"],
            model_name=model_name,
            dataset_name=dataset_name,
        )
        plot_training_curves(history, paths["clean_curves"])
        clean_model.load_state_dict(torch.load(paths["clean"], map_location=device))
        clean_model.eval()

    print("\nEvaluating clean model...")
    test(clean_model, test_loader, device)

    return clean_model


# ============================================================
# LOAD / TRAIN BADNET EXPLANATION MODEL
# ============================================================
def load_or_train_badnet(CONFIG, clean_model, train_loader, test_loader, device, model_name, dataset_name, paths):
    badnet_cfg = CONFIG["badnet"]
    model = copy.deepcopy(clean_model)

    if os.path.exists(paths["badnet"]) and not CONFIG["force_retrain"]:
        print(f"\nLoading BadNet explanation model from {paths['badnet']}")
        model.load_state_dict(torch.load(paths["badnet"], map_location=device))
        model.eval()
    else:
        print("\nTraining BadNet explanation attack...")
        model = train_explanation_badnet(
            model=model,
            orig_model=clean_model,
            train_loader=train_loader,
            device=device,
            num_epochs=badnet_cfg["epochs"],
            lambda_exp=badnet_cfg["lambda_exp"],
            poison_rate=badnet_cfg["poison_rate"],
            model_name=model_name,
            dataset_name=dataset_name,
        )
        model.eval()
        torch.save(model.state_dict(), paths["badnet"])

    print("\nEvaluating BadNet explanation model...")
    test(model, test_loader, device)
    return model


# ============================================================
# LOAD / TRAIN WANET EXPLANATION MODEL
# ============================================================
def load_or_train_wanet(CONFIG, clean_model, train_loader, test_loader, device, model_name, dataset_name, paths):
    wanet_cfg = CONFIG["wanet"]
    model = copy.deepcopy(clean_model)

    if (
        os.path.exists(paths["wanet"])
        and os.path.exists(paths["wanet_trigger"])
        and not CONFIG["force_retrain"]
    ):
        print(f"\nLoading WaNet explanation model from {paths['wanet']}")
        model.load_state_dict(torch.load(paths["wanet"], map_location=device))
        model.eval()

        # We still need the trigger object, so we reconstruct it by re-running
        # train_explanation_wanet logic is not ideal here unless you already have
        # a dedicated WaNet trigger class constructor somewhere.
        # If you have a dedicated trigger class, instantiate it here and load state.
        state = torch.load(paths["wanet_trigger"], map_location=device)

        # NOTE:
        # This assumes your train_explanation_wanet returns a trigger object with
        # .base_grid and .identity_grid. Replace this with your actual trigger class
        # if needed.
        _, wanet_trigger = train_explanation_wanet(
            model=copy.deepcopy(clean_model),
            orig_model=clean_model,
            train_loader=train_loader,
            device=device,
            num_epochs=0,  # dummy call if your code supports it; otherwise replace this logic
            lambda_exp=wanet_cfg["lambda_exp"],
            lambda_attack=wanet_cfg["lambda_attack"],
            rho_attack=wanet_cfg["rho_attack"],
            rho_noise=wanet_cfg["rho_noise"],
            model_name=model_name,
            dataset_name=dataset_name,
        )
        wanet_trigger.base_grid = state["base_grid"]
        wanet_trigger.identity_grid = state["identity_grid"]

    else:
        print("\nTraining WaNet explanation attack...")
        model, wanet_trigger = train_explanation_wanet(
            model=model,
            orig_model=clean_model,
            train_loader=train_loader,
            device=device,
            num_epochs=wanet_cfg["epochs"],
            lambda_exp=wanet_cfg["lambda_exp"],
            lambda_attack=wanet_cfg["lambda_attack"],
            rho_attack=wanet_cfg["rho_attack"],
            rho_noise=wanet_cfg["rho_noise"],
            model_name=model_name,
            dataset_name=dataset_name,
        )
        model.eval()
        torch.save(model.state_dict(), paths["wanet"])
        torch.save(
            {
                "base_grid": wanet_trigger.base_grid,
                "identity_grid": wanet_trigger.identity_grid,
            },
            paths["wanet_trigger"],
        )

    print("\nEvaluating WaNet explanation model...")
    test(model, test_loader, device)
    return model, wanet_trigger


# ============================================================
# LOAD / TRAIN GROND EXPLANATION MODEL
# ============================================================
def load_or_train_grond(CONFIG, clean_model, train_loader, test_loader, device, model_name, dataset_name, paths):
    grond_cfg = CONFIG["grond"]

    # ---- load / generate UPGD trigger ----
    if os.path.exists(paths["upgd_trigger"]) and not CONFIG["force_retrain"]:
        print(f"\nLoading Grond UPGD trigger from {paths['upgd_trigger']}")
        upgd_trigger = torch.load(paths["upgd_trigger"], map_location=device)
    else:
        print("\nGenerating Grond trigger...")
        cam_extractor = get_cam_extractor(clean_model, model_name)

        upgd_trigger = generate_upgd(
            model=clean_model,
            dataloader=train_loader,
            device=device,
            cam_extractor=cam_extractor,
            model_name=model_name,
        )

        cam_extractor.remove()
        torch.save(upgd_trigger, paths["upgd_trigger"])

    # ---- load / train explanation model ----
    model = copy.deepcopy(clean_model)

    if os.path.exists(paths["grond"]) and not CONFIG["force_retrain"]:
        print(f"\nLoading Grond explanation model from {paths['grond']}")
        model.load_state_dict(torch.load(paths["grond"], map_location=device))
        model.eval()
    else:
        print("\nTraining Grond explanation attack...")
        model = train_explanation_grond(
            model=model,
            orig_model=clean_model,
            train_loader=train_loader,
            upgd_trigger=upgd_trigger,
            device=device,
            num_epochs=grond_cfg["epochs"],
            lambda_exp=grond_cfg["lambda_exp"],
            lambda_attack=grond_cfg["lambda_attack"],
            poison_rate=grond_cfg["poison_rate"],
            model_name=model_name,
            dataset_name=dataset_name,
        )
        model.eval()
        torch.save(model.state_dict(), paths["grond"])

    print("\nEvaluating Grond explanation model...")
    test(model, test_loader, device)
    return model, upgd_trigger


# ============================================================
# STAGE A EVALUATION
# ============================================================
def run_stageA_evaluation(clean_model, expl_badnet_model, expl_wanet_model, expl_grond_model,
                          wanet_trigger, upgd_trigger, test_loader, device, model_name):
    print("\n=== Stage A Evaluation ===")

    def badnet_target_fn(h, w, b):
        return badnet_target_mask(h, w, b, device)

    def wanet_target_fn(h, w, b):
        return wanet_target_mask(wanet_trigger, h, w, b, device)

    def grond_target_fn(h, w, b):
        return upgd_target_mask(upgd_trigger, h, w, b, device)

    attacks = {
        "badnet": (expl_badnet_model, BadNetAttack(BadNetTrigger(size=20)), badnet_target_fn),
        "wanet": (expl_wanet_model, WaNetAttack(wanet_trigger), wanet_target_fn),
        "grond": (expl_grond_model, GrondAttack(upgd_trigger), grond_target_fn),
    }

    results = {}
    clean_model = clean_model.to(device)

    for name, (model, attack, target_fn) in attacks.items():
        print(f"\nEvaluating {name}...")
        model.to(device)

        results[name] = evaluate_explanations(
            model=model,
            orig_model=clean_model,
            test_loader=test_loader,
            attack_fn=attack,
            target_fn=target_fn,
            device=device,
            model_name=model_name,
        )

        model.to("cpu")
        torch.cuda.empty_cache()

    print("\n=== Explanation Results ===")
    for attack_name, metrics in results.items():
        print(f"\n{attack_name.upper()}")
        for metric_name, value in metrics.items():
            print(f"{metric_name:25s}: {value:.4f}")

    return results


# ============================================================
# STAGE A VISUALIZATION
# ============================================================
def run_stageA_visualization(expl_badnet_model, expl_wanet_model, expl_grond_model,
                             wanet_trigger, upgd_trigger, test_loader, classes, device, model_name):
    print("\n=== Stage A Visualization ===")

    visualize_clean_vs_triggered(
        expl_badnet_model,
        test_loader,
        classes,
        device,
        BadNetAttack(BadNetTrigger(size=20)),
        model_name=model_name,
        attack_name="BadNet",
    )

    visualize_clean_vs_triggered(
        expl_wanet_model,
        test_loader,
        classes,
        device,
        WaNetAttack(wanet_trigger),
        model_name=model_name,
        attack_name="WaNet",
    )

    visualize_clean_vs_triggered(
        expl_grond_model,
        test_loader,
        classes,
        device,
        GrondAttack(upgd_trigger),
        model_name=model_name,
        attack_name="Grond",
    )


# ============================================================
# BUILD STAGE B SETUP
# ============================================================
def build_stageB_setup(stageB_mode, expl_badnet_model, expl_wanet_model, expl_grond_model,
                       wanet_trigger, upgd_trigger, device, CONFIG):
    """
    Returns:
        base_expl_model, attack_pred, attack_exp, target_fn, target_label
    """

    if stageB_mode == "badnet+badnet":
        base_expl_model = expl_badnet_model

        trigger_pred = BadNetTrigger(size=20, position="bottom-right")
        trigger_exp = BadNetTrigger(size=20, position="upper-left")

        attack_pred = lambda x: apply_badnet(x, trigger_pred)
        attack_exp = lambda x: apply_badnet(x, trigger_exp)

        target_fn = lambda h, w, b: badnet_target_mask(h, w, b, device)
        target_label = CONFIG["badnet"]["stageB"]["target_label"]

    elif stageB_mode == "wanet+badnet":
        base_expl_model = expl_wanet_model

        trigger_pred = BadNetTrigger(size=20, position="bottom-right")
        trigger_exp = wanet_trigger

        attack_pred = lambda x: apply_badnet(x, trigger_pred)
        attack_exp = lambda x: trigger_exp.warp(x.clone())

        target_fn = lambda h, w, b: wanet_target_mask(trigger_exp, h, w, b, device)
        target_label = CONFIG["badnet"]["stageB"]["target_label"]

    elif stageB_mode == "grond+wanet":
        base_expl_model = expl_grond_model

        trigger_pred = wanet_trigger
        trigger_exp = upgd_trigger

        attack_pred = lambda x: trigger_pred.warp(x.clone())
        attack_exp = GrondAttack(trigger_exp)

        target_fn = lambda h, w, b: upgd_target_mask(trigger_exp, h, w, b, device)
        target_label = CONFIG["wanet"]["stageB"]["target_label"]

    else:
        raise ValueError(f"Unknown stageB_mode: {stageB_mode}")

    return base_expl_model, attack_pred, attack_exp, target_fn, target_label


# ============================================================
# LOAD / TRAIN STAGE B MODEL
# ============================================================
def load_or_train_stageB_model(CONFIG, clean_model, base_expl_model, train_loader, test_loader,
                               device, model_name, dataset_name, stageB_mode,
                               attack_pred, attack_exp, target_fn, paths):
    pred_model = copy.deepcopy(base_expl_model)
    pred_model = freeze_model(pred_model, model_name)

    # choose config depending on prediction attack
    if stageB_mode in ["badnet+badnet", "wanet+badnet"]:
        stageB_cfg = CONFIG["badnet"]["stageB"]
    elif stageB_mode == "grond+wanet":
        stageB_cfg = CONFIG["wanet"]["stageB"]
    else:
        raise ValueError(f"Unknown stageB_mode: {stageB_mode}")

    if os.path.exists(paths["stageB"]) and not CONFIG["force_retrain"]:
        print(f"\nLoading Stage B model from {paths['stageB']}")
        pred_model.load_state_dict(torch.load(paths["stageB"], map_location=device))
        pred_model.eval()
    else:
        print(f"\nTraining Stage B model ({stageB_mode})...")
        pred_model = train_prediction_badnet(
            model=pred_model,
            orig_model=clean_model,
            train_loader=train_loader,
            attack_pred=attack_pred,
            attack_exp=attack_exp,
            target_fn=target_fn,
            device=device,
            config=stageB_cfg,
            model_name=model_name,
            dataset_name=dataset_name,
        )
        pred_model.eval()
        torch.save(pred_model.state_dict(), paths["stageB"])

    print("\nEvaluating Stage B model...")
    test(pred_model, test_loader, device)

    return pred_model


# ============================================================
# STAGE B EVALUATION
# ============================================================
def run_stageB_evaluation(pred_model, clean_model, test_loader, attack_pred, attack_exp,
                          target_fn, target_label, device, model_name):
    print("\n=== Stage B Evaluation ===")

    print("\n--- Stage B: Prediction Attack ---")
    calculate_ASR(pred_model, test_loader, attack_pred, device, target_label=target_label)

    print("\n--- Stage B: Explanation Preservation (B) ---")
    evaluate_explanation_preservation(pred_model, test_loader, attack_pred, device, model_name)

    print("\n--- Stage B: Explanation Attack (A) ---")
    evaluate_explanation_consistency(
        pred_model,
        clean_model,
        test_loader,
        attack_exp,
        target_fn,
        device,
        model_name,
    )

    print("\n--- Stage B: Combined Behavior (A+B) ---")
    evaluate_prediction_AB(pred_model, test_loader, attack_exp, attack_pred, device, target_label=target_label)
    evaluate_explanation_AB(pred_model, test_loader, attack_exp, attack_pred, target_fn, device, model_name)


# ============================================================
# STAGE B VISUALIZATION
# ============================================================
def run_stageB_visualization(pred_model, test_loader, attack_exp, attack_pred, classes,
                             device, model_name, save_path):
    print("\n=== Stage B Visualization ===")
    visualize_AB(
        pred_model,
        test_loader,
        attack_exp,
        attack_pred,
        classes,
        device,
        model_name=model_name,
        num_images=4,
        save_path=save_path,
    )


# ============================================================
# MAIN
# ============================================================
def main(CONFIG):
    setup_seed(CONFIG.get("seed", 42))
    device = get_device()
    enable_safe_transformer_kernels()

    os.makedirs("models", exist_ok=True)

    model_name = CONFIG["model"].lower()
    dataset_name = CONFIG["dataset"].lower()
    stageB_mode = CONFIG["stageB_mode"]
    run_stage = CONFIG.get("run_stage", "full")  # clean / stageA / stageB / full

    paths = get_paths(model_name, dataset_name, stageB_mode)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    train_loader, val_loader, test_loader, classes = get_dataloaders(model_name, dataset_name)
    n_classes = len(classes)

    # --------------------------------------------------------
    # Clean model (always needed)
    # --------------------------------------------------------
    clean_model = load_or_train_clean_model(
        CONFIG, model_name, dataset_name,
        train_loader, val_loader, test_loader,
        n_classes, device, paths
    )

    # Safe Grad-CAM call
    try:
        visualize_gradcam_batch(clean_model, test_loader, classes, device)
    except TypeError:
        # If your function *does* require model_name, keep this fallback
        visualize_gradcam_batch(clean_model, test_loader, classes, device, model_name=model_name)

    if run_stage == "clean":
        print("\nFinished clean-model run.")
        return

    # --------------------------------------------------------
    # Stage A models
    # --------------------------------------------------------
    expl_badnet_model = load_or_train_badnet(
        CONFIG, clean_model, train_loader, test_loader,
        device, model_name, dataset_name, paths
    )

    expl_wanet_model, wanet_trigger = load_or_train_wanet(
        CONFIG, clean_model, train_loader, test_loader,
        device, model_name, dataset_name, paths
    )

    expl_grond_model, upgd_trigger = load_or_train_grond(
        CONFIG, clean_model, train_loader, test_loader,
        device, model_name, dataset_name, paths
    )

    if run_stage in ["stageA", "full", "stageB"]:
        run_stageA_evaluation(
            clean_model=clean_model,
            expl_badnet_model=expl_badnet_model,
            expl_wanet_model=expl_wanet_model,
            expl_grond_model=expl_grond_model,
            wanet_trigger=wanet_trigger,
            upgd_trigger=upgd_trigger,
            test_loader=test_loader,
            device=device,
            model_name=model_name,
        )

        run_stageA_visualization(
            expl_badnet_model=expl_badnet_model,
            expl_wanet_model=expl_wanet_model,
            expl_grond_model=expl_grond_model,
            wanet_trigger=wanet_trigger,
            upgd_trigger=upgd_trigger,
            test_loader=test_loader,
            classes=classes,
            device=device,
            model_name=model_name,
        )

    if run_stage == "stageA":
        print("\nFinished Stage A run.")
        return

    # --------------------------------------------------------
    # Stage B setup
    # --------------------------------------------------------
    base_expl_model, attack_pred, attack_exp, target_fn, target_label = build_stageB_setup(
        stageB_mode=stageB_mode,
        expl_badnet_model=expl_badnet_model,
        expl_wanet_model=expl_wanet_model,
        expl_grond_model=expl_grond_model,
        wanet_trigger=wanet_trigger,
        upgd_trigger=upgd_trigger,
        device=device,
        CONFIG=CONFIG,
    )

    pred_model = load_or_train_stageB_model(
        CONFIG=CONFIG,
        clean_model=clean_model,
        base_expl_model=base_expl_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        model_name=model_name,
        dataset_name=dataset_name,
        stageB_mode=stageB_mode,
        attack_pred=attack_pred,
        attack_exp=attack_exp,
        target_fn=target_fn,
        paths=paths,
    )

    run_stageB_evaluation(
        pred_model=pred_model,
        clean_model=clean_model,
        test_loader=test_loader,
        attack_pred=attack_pred,
        attack_exp=attack_exp,
        target_fn=target_fn,
        target_label=target_label,
        device=device,
        model_name=model_name,
    )

    run_stageB_visualization(
        pred_model=pred_model,
        test_loader=test_loader,
        attack_exp=attack_exp,
        attack_pred=attack_pred,
        classes=classes,
        device=device,
        model_name=model_name,
        save_path=paths["stageB_vis"],
    )

    print("\nFinished full pipeline.")


# ============================================================
# CONFIG
# ============================================================
if __name__ == "__main__":
    CONFIG = {
        "seed": 42,
        "model": "deit_small",
        "dataset": "cifar10",

        # clean training
        "epochs": 100,

        # what to run: "clean" / "stageA" / "stageB" / "full"
        "run_stage": "full",

        # if False: load existing models/triggers when found
        # if True: retrain / regenerate everything
        "force_retrain": False,

        # choose Stage B composition
        "stageB_mode": "badnet+badnet",   # "badnet+badnet" / "wanet+badnet" / "grond+wanet"

        "badnet": {
            # Stage A explanation
            "epochs": 30,
            "lambda_exp": 0.4,
            "poison_rate": 0.2,

            # Stage B prediction
            "stageB": {
                "epochs": 8,
                "poison_rate": 0.1,
                "lambda_preserve": 0.3,
                "lambda_A": 0.3,
                "lambda_AB": 0.2,
                "target_label": 0,
            },
        },

        "wanet": {
            # Stage A explanation
            "epochs": 30,
            "lambda_exp": 0.4,
            "lambda_attack": 3.0,
            "rho_attack": 0.2,
            "rho_noise": 0.1,

            # Stage B prediction
            "stageB": {
                "epochs": 8,
                "poison_rate": 0.1,
                "lambda_preserve": 0.3,
                "lambda_A": 0.3,
                "lambda_AB": 0.2,
                "target_label": 0,
            },
        },

        "grond": {
            "epochs": 30,
            "lambda_exp": 0.4,
            "lambda_attack": 3.0,
            "poison_rate": 0.2,
        },
    }

    main(CONFIG)