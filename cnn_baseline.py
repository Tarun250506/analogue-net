
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

MANIFEST_PATH = RESULTS_DIR / "shared_split_manifest.csv"
MODEL_PATH = MODEL_DIR / "analoguennet_mobilenetv3.pth"
REPORT_PATH = RESULTS_DIR / "mobilenetv3_test_report.txt"

SEED = 42
EPOCHS = 15
BATCH_SIZE = 4
LEARNING_RATE = 1e-3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# --------------------------------------------------
# 2. Dataset
# --------------------------------------------------

class PaneerDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]
        image_path = ROOT / row["path"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        image = self.transform(image)
        label = int(row["label"])

        return image, label


# --------------------------------------------------
# 3. Training and evaluation
# --------------------------------------------------

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Shared split missing. Run python create_split.py first."
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    required = {"path", "sample_id", "label", "split"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest must contain: {required}")

    if manifest["path"].duplicated().any():
        raise ValueError("Duplicate image paths in manifest.")

    train_df = manifest[manifest["split"] == "train"].copy()
    val_df = manifest[manifest["split"] == "validation"].copy()
    test_df = manifest[manifest["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Train, validation and test splits are required.")

    for relative_path in manifest["path"]:
        if not (ROOT / relative_path).is_file():
            raise FileNotFoundError(f"Missing image: {relative_path}")

    print(f"Device: {DEVICE}")
    print(
        f"Samples: {len(train_df)} train | "
        f"{len(val_df)} validation | {len(test_df)} test"
    )

    # ImageNet normalization, matching the ViT experiment.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    train_loader = DataLoader(
        PaneerDataset(train_df, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        PaneerDataset(val_df, eval_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        PaneerDataset(test_df, eval_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------
    # MobileNetV3-Small pretrained CNN
    # --------------------------------------------------

    model = timm.create_model(
        "mobilenetv3_small_100",
        pretrained=True,
        num_classes=2,
    ).to(DEVICE)

    # Freeze the pretrained feature extractor.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # In timm's MobileNetV3 models, get_classifier()
    # returns the final classification layer.
    classifier = model.get_classifier()

    for parameter in classifier.parameters():
        parameter.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    # --------------------------------------------------
    # Train and select checkpoint using validation loss
    # --------------------------------------------------

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss_total = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * images.size(0)

        train_loss = train_loss_total / len(train_df)

        model.eval()
        val_loss_total = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_total += loss.item() * images.size(0)

        val_loss = val_loss_total / len(val_df)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"validation loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Saved checkpoint: {MODEL_PATH}")

    # --------------------------------------------------
    # Evaluate once on the held-out test set
    # --------------------------------------------------

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)

            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["untreated", "starch_treated"],
        zero_division=0,
    )

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\nMobileNetV3-Small — SHARED TEST SET")
    print(report)
    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)

    with open(REPORT_PATH, "w") as file:
        file.write("MobileNetV3-Small evaluation\n\n")
        file.write(report)
        file.write("\nConfusion matrix:\n")
        file.write(str(matrix))
        file.write("\n")

    print(f"\nSaved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
