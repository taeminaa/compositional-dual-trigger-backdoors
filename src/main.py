import torch
import copy
import os
import argparse

from objectives.evaluation.stageA_metrics import evaluate_explanations
from objectives.evaluation.stageA_visualization import visualize_clean_vs_triggered
from objectives.evaluation.stageB_visualization import visualize_joint_triggers
from utils.utils import normalize, denormalize, enable_safe_transformer_kernels, get_cam_extractor, freeze_model, get_paths
from objectives.attacks.badnet import train_explanation_badnet, apply_badnet, badnet_target_mask, train_prediction_badnet, BadNetTrigger
from objectives.attacks.wanet import train_explanation_wanet, wanet_target_mask, train_prediction_wanet, ExplanationWaNet
from objectives.attacks.grond import train_explanation_grond, generate_upgd, upgd_target_mask
from training.train_clean_model import get_device,setup_seed,get_dataloaders,get_clean_model,train_clean_model,test
from objectives.evaluation.stageB_metrics import calculate_ASR, evaluate_explanation_preservation, evaluate_explanation_retention, evaluate_prediction_AB, evaluate_explanation_AB



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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Explanation-aware and compositional backdoor training."
    )

    parser.add_argument(
        "--model",
        default="mobilenetv2",
        choices=[
            "resnet18",
            "vgg16",
            "mobilenetv2",
            "tiny_vit",
            "deit_small",
        ],
        help="Model architecture.",
    )

    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=[
            "cifar10",
            "cifar100",
            "tiny_imagenet",
        ],
        help="Dataset.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Clean model training epochs.",
    )

    parser.add_argument(
        "--stageB_mode",
        default="badnet+badnet",
        choices=[
            "badnet+badnet",
            "wanet+badnet",
            "grond+wanet",
        ],
        help="Trigger composition.",
    )

    return parser.parse_args()

def get_default_config(dataset, clean_epochs):

    """
    Returns the default hyperparameters.
    """

    stageA_epochs = 45 if dataset == "tiny_imagenet" else 30
    stageB_epochs = 15 if dataset == "tiny_imagenet" else 8

    return {
        "epochs": clean_epochs,

        "badnet": {
            "epochs": stageA_epochs,
            "lambda_exp": 0.4,
            "poison_rate": 0.2,
        },

        "wanet": {
            "epochs": stageA_epochs,
            "lambda_exp": 0.4,
            "lambda_attack": 3.0,
            "rho_attack": 0.2,
            "rho_noise": 0.1,
        },

        "grond": {
            "epochs": stageA_epochs,
            "lambda_exp": 0.4,
            "lambda_attack": 3.0,
            "poison_rate": 0.2,
        },

        "stageB": {
            "epochs": stageB_epochs,
            "poison_rate": 0.1,
            "lambda_preserve": 0.3,
            "lambda_A": 0.3,
            "lambda_AB": 0.2,
            "target_label": 0,
        },
    }



def main(CONFIG):

    setup_seed(42)
    device = get_device()
    enable_safe_transformer_kernels()
    os.makedirs("models", exist_ok=True)

    model_name = CONFIG["model"].lower()
    dataset_name = CONFIG["dataset"].lower()

    paths = get_paths(model_name, dataset_name, CONFIG["stageB_mode"])
    train_dataloader, val_dataloader, test_dataloader, classes = get_dataloaders(model_name, dataset_name)
    n_classes = len(classes)
    clean_model = get_clean_model(model_name, n_classes, device=device)

    if os.path.exists(paths["clean"]):
        print("\nLoading clean model...")
        clean_model.load_state_dict(torch.load(paths["clean"], map_location=device))

    else:
        print("\n Training Clean Model...")
        clean_model, history = train_clean_model(
            clean_model,
            train_dataloader,
            val_dataloader,
            device,
            epochs=CONFIG["epochs"],
            model_name=model_name,
            dataset_name= dataset_name,
            paths=paths
        )

    clean_model.eval()
    print("\n Evaluating Clean model...")
    test(clean_model, test_dataloader, device)

    # ==============================
    # BADNET TRAINING
    # ==============================

    expl_badnet_model = copy.deepcopy(clean_model)
    if os.path.exists(paths["badnet"]):
        print("\nLoading BadNet model...")
        expl_badnet_model.load_state_dict(
            torch.load(paths["badnet"], map_location=device)
        )

    else:
        print("\n Training BadNet Explanation Attack...")
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
        torch.save(expl_badnet_model.state_dict(), paths["badnet"])

    expl_badnet_model.eval()
    print("\n Evaluating BadNet model...")
    test(expl_badnet_model, test_dataloader, device)

    # ==============================
    # WANET TRAINING
    # ==============================    


    expl_wanet_model = copy.deepcopy(clean_model)
    if os.path.exists(paths["wanet"]) and os.path.exists(paths["wanet_trigger"]):

        print("\nLoading WaNet model...")
        expl_wanet_model.load_state_dict(
            torch.load(paths["wanet"], map_location=device)
        )

        state = torch.load(paths["wanet_trigger"], map_location=device)
        wanet_trigger = ExplanationWaNet(image_size=(224,224),device=device)
        wanet_trigger.base_grid = state["base_grid"]
        wanet_trigger.identity_grid = state["identity_grid"]

    else: 
        print("\n Training WaNet Explanation Attack...")  
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
        torch.save(expl_wanet_model.state_dict(), paths["wanet"])
        torch.save({"base_grid": wanet_trigger.base_grid, "identity_grid": wanet_trigger.identity_grid}, paths["wanet_trigger"])

    expl_wanet_model.eval()
    print("\n Evaluating WaNet model...")
    test(expl_wanet_model, test_dataloader, device)

    # ==============================
    # Grond TRAINING
    # ==============================
    print("\n Generating Grond trigger...")

    if os.path.exists(paths["upgd_trigger"]):
        upgd_trigger = torch.load(paths["upgd_trigger"], map_location=device)
    else:
        cam = get_cam_extractor(clean_model, model_name)
       
        upgd_trigger = generate_upgd(
            model=clean_model,
            dataloader=train_dataloader,
            device=device,
            cam_extractor=cam,
            model_name=model_name,
        )

        cam.remove()
        torch.save(upgd_trigger, paths["upgd_trigger"])

    expl_grond_model = copy.deepcopy(clean_model)

    if os.path.exists(paths["grond"]):
        print("\nLoading Grond model...")
        expl_grond_model.load_state_dict(
            torch.load(paths["grond"], map_location=device)
        )

    else:
        print("\n Training Grond Explanation Attack...")
        grond_cfg = CONFIG["grond"]

        expl_grond_model = train_explanation_grond(
            model=expl_grond_model,
            orig_model=clean_model,
            train_loader=train_dataloader,
            upgd_trigger=upgd_trigger,
            device=device,
            num_epochs=grond_cfg["epochs"],
            lambda_exp=grond_cfg["lambda_exp"],
            lambda_attack=grond_cfg["lambda_attack"],
            poison_rate=grond_cfg["poison_rate"],
            model_name=model_name,
            dataset_name=dataset_name  
        )
        torch.save(expl_grond_model.state_dict(), paths["grond"])

    expl_grond_model.eval()
    print("\n Evaluating Grond model...")
    test(expl_grond_model, test_dataloader, device)

    # ==============================
    # EVALUATION stage A
    # ==============================
    print("\n Overall Evaluation...")

    target_functions = {
        "badnet": lambda h, w, b: badnet_target_mask(h, w, b, device),
        "wanet": lambda h, w, b: wanet_target_mask(wanet_trigger, h, w, b, device),
        "grond": lambda h, w, b: upgd_target_mask(upgd_trigger, h, w, b, device)
    }

    attacks = {
        "badnet": (expl_badnet_model, BadNetAttack(BadNetTrigger(size=20))),
        "wanet": (expl_wanet_model, WaNetAttack(wanet_trigger)),
        "grond": (expl_grond_model, GrondAttack(upgd_trigger))
    }

    results = {}
    clean_model.to(device)

    for attack_name, (model, attack) in attacks.items():
        print(f"\nEvaluating {attack_name}...")
        model.to(device)

        results[attack_name] = evaluate_explanations(
            model=model,
            clean_model=clean_model,
            dataloader=test_dataloader,
            attack=attack,
            target_fn=target_functions[attack_name],
            device=device,
            model_name=model_name,
        )

        model.to("cpu")
        torch.cuda.empty_cache()

    print("\n=== Explanation Results ===")

    for attack_name, metrics in results.items():
        print(f"\n{attack_name.upper()}")
        for metric, value in metrics.items():
            print(f"{metric:25s}: {value:.4f}")

    # ==============================
    # VISUALIZATION stage A
    # ==============================
    for attack_name, (model, attack) in attacks.items():

        visualize_clean_vs_triggered(
            model=model,
            dataloader=test_dataloader,
            classes=classes,
            device=device,
            attack=attack,
            model_name=model_name,
            dataset_name=dataset_name,
            attack_name=attack_name.capitalize(),
        )

    # ==============================
    # Set up STAGE B
    # ==============================

    stageB_mode = CONFIG["stageB_mode"]
    stageB_config = CONFIG["stageB"]
    target_label = stageB_config["target_label"]

    if stageB_mode == "badnet+badnet":
        stageB_model = copy.deepcopy(expl_badnet_model) 

        trigger_pred = BadNetTrigger(size=20, position="bottom-right")
        trigger_exp  = BadNetTrigger(size=20, position="upper-left")

        attack_pred = lambda x: apply_badnet(x, trigger_pred)
        attack_exp  = lambda x: apply_badnet(x, trigger_exp)

        target_fn = lambda h, w, b: badnet_target_mask(h, w, b, device)
        trainer = train_prediction_badnet
       

    elif stageB_mode == "wanet+badnet":

        stageB_model = copy.deepcopy(expl_wanet_model) 

        trigger_pred = BadNetTrigger(size=20, position="bottom-right")
        trigger_exp = wanet_trigger

        attack_pred = lambda x: apply_badnet(x, trigger_pred)
        attack_exp  = lambda x: trigger_exp.warp(x.clone())

        target_fn = lambda h, w, b: wanet_target_mask(trigger_exp, h, w, b, device)
        trainer = train_prediction_badnet
    

    elif stageB_mode == "grond+wanet":
         
        stageB_model = copy.deepcopy(expl_grond_model) 

        trigger_pred = wanet_trigger
        trigger_exp = upgd_trigger

        attack_pred = lambda x: trigger_pred.warp(x.clone())
        attack_exp  = GrondAttack(trigger_exp)

        target_fn = lambda h, w, b: upgd_target_mask(trigger_exp, h, w, b, device) 
        trainer = train_prediction_wanet

    else:
        raise ValueError(f"Unknown stageB_mode: {stageB_mode}")

    # ============================================================
    # Stage B Training
    # ============================================================
    if os.path.exists(paths["stageB"]):
    
            print("\nLoading Stage B model...")
            stageB_model.load_state_dict(
                torch.load(paths["stageB"], map_location=device)
            )
    else:
        print("\nTraining Stage B...")
        stageB_model = freeze_model(stageB_model,model_name)

        stageB_model = trainer(
            model=stageB_model,
            orig_model=clean_model,
            train_loader=train_dataloader,
            attack_pred=attack_pred,
            attack_exp=attack_exp,
            target_fn=target_fn,
            device=device,
            config=stageB_config,
            model_name=model_name,
            dataset_name=dataset_name
        )
        torch.save(stageB_model.state_dict(), paths["stageB"])

    stageB_model.eval()
    print("\n Evaluating A+B model...")
    test(stageB_model, test_dataloader, device)


    # ==============================
    # EVALUATION stage B
    # ==============================
    print("\n Stage B Evaluation...")

    print("\n--- Stage B: Prediction Attack ---")
    calculate_ASR(stageB_model, test_dataloader, attack_pred, device, target_label= target_label)

    print("\n--- Stage B: Explanation Preservation  ---")
    evaluate_explanation_preservation(stageB_model,test_dataloader, attack_pred, device, model_name)

    print("\n--- Stage B: Explanation Retention ---")
    evaluate_explanation_retention(stageB_model,clean_model,test_dataloader, attack_exp, target_fn, device, model_name)

    print("\n--- Stage B: Combined Behavior (A+B) ---")
    evaluate_prediction_AB(stageB_model,test_dataloader,attack_exp,attack_pred,device,target_label= target_label)
    evaluate_explanation_AB(stageB_model,test_dataloader,attack_exp,attack_pred,target_fn, device,model_name)

    visualize_joint_triggers(stageB_model, test_dataloader, attack_exp, attack_pred, classes, device, model_name, dataset_name, attack_name = stageB_mode.replace("+", "_"))


if __name__ == "__main__":

    args = parse_args()

    CONFIG = {
        "model": args.model,
        "dataset": args.dataset,
        "stageB_mode": args.stageB_mode,
    }

    CONFIG.update(get_default_config(args.dataset, args.epochs))

    print("=" * 50)
    print("Configuration")
    print("=" * 50)
    print(f"Model:       {CONFIG['model']}")
    print(f"Dataset:     {CONFIG['dataset']}")
    print(f"Clean epochs:{CONFIG['epochs']}")
    print(f"Stage A epochs: {CONFIG['badnet']['epochs']}")
    print(f"Stage B epochs: {CONFIG['stageB']['epochs']}")
    print(f"Composition:     {CONFIG['stageB_mode']}")
    print("=" * 50)

    main(CONFIG)
