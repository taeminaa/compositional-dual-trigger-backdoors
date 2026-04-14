import torch
import copy
import os

from src.objectives.explanation.attacks.badnet import (
    train_explanation_badnet,
    apply_badnet,
    badnet_target_mask,
    BadNetTrigger
)
from src.objectives.explanation.attacks.wanet import (
    train_explanation_wanet,
    wanet_target_mask
)
from src.objectives.explanation.attacks.grond import (
    train_explanation_grond,
    generate_all_upgd,
    upgd_target_mask
)
from src.objectives.explanation.gradcam.train_gradcam import TrainableGradCAMPP
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
from utils.utils import normalize, denormalize

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



def main():

    setup_seed(42)
    device = get_device()
    os.makedirs("models", exist_ok=True)

    train_dataloader, val_dataloader, test_dataloader, classes = get_dataloaders()
    clean_model = get_clean_model(n_classes=100, device=device)

    print("\n Training Clean Model...")
    clean_model, history = train_clean_model(
        clean_model,
        train_dataloader,
        val_dataloader,
        device,
        epochs=100,
        save_path="models/clean.pth"
    )

    plot_training_curves(history, "models/training_curves.png")
    print("\n Evaluating Clean model...")
    test(clean_model, test_dataloader, device)

    # ==============================
    # BADNET TRAINING
    # ==============================
    print("\n Training BadNet Explanation Attack...")

    expl_badnet_model = copy.deepcopy(clean_model)

    expl_badnet_model = train_explanation_badnet(
        model=expl_badnet_model,
        orig_model=clean_model,
        train_loader=train_dataloader,
        device=device,
        num_epochs=30,
        lambda_exp=0.4,
        poison_rate=0.2,
        lr=3e-4
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

    expl_wanet_model, wanet_trigger = train_explanation_wanet(
        model=expl_wanet_model,
        orig_model=clean_model,
        train_loader=train_dataloader,
        device=device,
        num_epochs=30,
        lambda_exp=0.4,
        lambda_attack=3.0,
        rho_attack=0.2,
        rho_noise=0.1,
        lr=3e-4,
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
        upgd_triggers = torch.load("models/upgd_triggers.pth", map_location=device)
    else:
        cam_extractor = TrainableGradCAMPP(clean_model, clean_model.features[-1])
        upgd_triggers = generate_all_upgd(
            model=clean_model,
            dataloader=train_dataloader,
            device=device,
            cam_extractor=cam_extractor,
            num_classes= 100
        )

        cam_extractor.remove()
        torch.save(upgd_triggers, "models/upgd_triggers.pth")

    avg_upgd_trigger = torch.stack(upgd_triggers).mean(dim=0)

    print("\n Training Grond Explanation Attack...")

    expl_grond_model = copy.deepcopy(clean_model)

    expl_grond_model = train_explanation_grond(
        model=expl_grond_model,
        orig_model=clean_model,
        train_loader=train_dataloader,
        upgd_triggers=upgd_triggers,
        device=device,
        num_epochs=30,
        lambda_exp=0.4,
        lambda_attack=3.0,
        poison_rate=0.2,
        lr=3e-4
    )

    expl_grond_model.eval()
    torch.save(expl_grond_model.state_dict(), "models/grond.pth")
    print("\n Evaluating Grond model...")
    test(expl_grond_model, test_dataloader, device)

    # ==============================
    # EVALUATION
    # ==============================
    print("\n Overall Evaluation...")

    def badnet_target_fn(h, w, b):
        return badnet_target_mask(h, w, b, device)

    def wanet_target_fn(h, w, b):
        return wanet_target_mask(wanet_trigger, h, w, b, device)

    def grond_target_fn(h, w, b):
        return upgd_target_mask(avg_upgd_trigger, h, w, b, device)

    ATTACKS = {
        "badnet": (expl_badnet_model, BadNetAttack(BadNetTrigger(size=20)), badnet_target_fn),
        "wanet": (expl_wanet_model, WaNetAttack(wanet_trigger), wanet_target_fn),
        "grond": (expl_grond_model, GrondAttack(avg_upgd_trigger), grond_target_fn),
    }

    results = {}

    for name, (model, attack, target_fn) in ATTACKS.items():
        print(f"\nEvaluating {name}...")
        model.to(device)

        results[name] = evaluate_explanations(
            model, clean_model, test_dataloader, attack, target_fn, device
        )

        model.to("cpu")
        torch.cuda.empty_cache()

    print("\n=== Explanation Results ===")
    for k, v in results.items():
        print(f"\n{k.upper()}")
        for metric, val in v.items():
            print(f"{metric:25s}: {val:.4f}")

    # ==============================
    # VISUALIZATION
    # ==============================
    visualize_clean_vs_triggered(
        expl_badnet_model, test_dataloader, classes, device,
        BadNetAttack(BadNetTrigger(size=20)), "BadNet"
    )

    visualize_clean_vs_triggered(
        expl_wanet_model, test_dataloader, classes, device,
        WaNetAttack(wanet_trigger), "WaNet"
    )

    visualize_clean_vs_triggered(
        expl_grond_model, test_dataloader, classes, device,
        GrondAttack(avg_upgd_trigger), "GROND"
    )


if __name__ == "__main__":
    main()



# ==============================
# Load backdoor models
# ==============================

# expl_badnet_model = models.vgg16_bn(weights=None)
# expl_badnet_model.classifier[6] = nn.Linear(expl_badnet_model.classifier[6].in_features, len(classes))
# expl_badnet_model = expl_badnet_model.to(device)
# expl_badnet_model.load_state_dict(
#     torch.load(f"{PROJECT_DIR}/vgg16_cifar100_expl_badnet.pth", map_location=device)
# )
# expl_badnet_model.eval()

# wanet_trigger = ExplanationWaNet(image_size=(224, 224),device=device)
# state = torch.load(f"{PROJECT_DIR}/vgg16_wanet_trigger.pth", map_location=device)
# wanet_trigger.base_grid = state["base_grid"]
# wanet_trigger.identity_grid = state["identity_grid"]
# expl_wanet_model = models.vgg16_bn(weights=None)
# expl_wanet_model.classifier[6] = nn.Linear(expl_wanet_model.classifier[6].in_features, len(classes))
# expl_wanet_model = expl_wanet_model.to(device)
# expl_wanet_model.load_state_dict(
#     torch.load(f"{PROJECT_DIR}/vgg16_cifar100_expl_wanet.pth", map_location=device)
# )
# expl_wanet_model.eval()



# upgd_trigger = torch.load(f"{PROJECT_DIR}/vgg16_upgd_trigger.pth",map_location=device)
# expl_grond_model = models.vgg16_bn(weights=None)
# expl_grond_model.classifier[6] = nn.Linear(expl_grond_model.classifier[6].in_features, len(classes))
# expl_grond_model = expl_grond_model.to(device)
# expl_grond_model.load_state_dict(
#     torch.load(f"{PROJECT_DIR}/vgg16_cifar100_expl_grond.pth", map_location=device)
# )
# expl_grond_model.eval()

