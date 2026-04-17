import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim 
import timm

import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from src.config.model_config import MODEL_CONFIG

# ==============================
# Device
# ==============================
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    return device

# ==============================
# Seed for reproducibility
# ==============================
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================
# Data
# ==============================
def get_dataloaders(model_name, dataset_name, batch_size=64, num_workers=2, seed=42):
    model_name = model_name.lower()
    cfg = MODEL_CONFIG[model_name]

    if cfg["type"] == "vit":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    elif cfg["type"] == "cnn":
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=8),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    if dataset_name == "cifar10":
        Dataset = datasets.CIFAR10
    elif dataset_name == "cifar100":
        Dataset = datasets.CIFAR100
       
    else:
        raise ValueError("Unsupported dataset")

    full_dataset = Dataset(root="./data", download=True, train=True)
    testset = Dataset(root="./data", download=True, train=False, transform=test_transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform

    def _init_fn(worker_id):
        np.random.seed(seed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, worker_init_fn=_init_fn)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, worker_init_fn=_init_fn)

    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, worker_init_fn=_init_fn)

    print("Train:", len(train_dataset))
    print("Val:", len(val_dataset))
    print("Test:", len(testset))

    return train_dataloader, val_dataloader, test_dataloader, testset.classes


# ==============================
# Model
# ==============================
def get_clean_model(model_name, n_classes, device):

    model_name = model_name.lower()
    # ======================
    # CNN MODELS
    # ======================
    if model_name == "vgg16":
        model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)

    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)

    # ======================
    # ViT MODELS
    # ======================
    elif model_name == "tiny_vit":
        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True,
            num_classes=n_classes
        )

    elif model_name == "deit_small":
        model = timm.create_model(
            "deit_small_patch16_224",
            pretrained=True,
            num_classes=n_classes
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)

# ==============================
# Optimizer
# ==============================
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

# ==============================
# Scheduler
# ==============================    
def get_scheduler(optimizer, model_name, epochs):
    model_name = model_name.lower()
    cfg = MODEL_CONFIG[model_name]

    if cfg["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

    elif cfg["scheduler"] == "warmup_cosine":

        warmup_epochs = 10

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1 + np.cos(
                np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)
            ))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ==============================
# Accuracy Function
# ==============================
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return (preds == labels).float().mean()

# ==============================
# Validation
# ==============================
def validate(model, loader, criterion, device):

    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            total_correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), total_correct / total


# ==============================
# Training
# ==============================
def train_clean_model(model, train_dataloader, val_dataloader, device, epochs, save_path, model_name, dataset_name):

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, model_name, dataset_name)
    scheduler = get_scheduler(optimizer, model_name, epochs)

    best_acc = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(epochs):

        model.train()

        running_loss = 0
        running_correct = 0
        total = 0

        print(f"\nEpoch {epoch+1}/{epochs}")

        for images, labels in tqdm(train_dataloader):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_dataloader)
        train_acc = running_correct / total

        val_loss, val_acc = validate(model, val_dataloader, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    return model, history


# ==============================
# Train Curves
# ==============================
def plot_training_curves(history, save_path):

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ==============================
# Load Best Model
# ==============================
# def load_clean_model(path, n_classes, device):
#     model = models.vgg16_bn(weights=None)
#     model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)
#     model.load_state_dict(torch.load(path, map_location=device))
#     return model.to(device).eval()


# ==============================
# Test Accuracy
# ==============================
def test(model, test_loader, device):

    model.eval()
    correct, total = 0, 0
   

    with torch.no_grad():
        for images, labels in tqdm(test_loader):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print("Test Accuracy:", acc)

    return acc


