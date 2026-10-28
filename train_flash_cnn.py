
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MANIFEST_PATH = ROOT / "results" / "shared_split_manifest.csv"

# Use v2 preprocessing outputs
FLASH_DIR = ROOT / "results" / "flash_roi_v2"

# Use the v2 preprocessing report
PIPELINE_REPORT_PATH = (
    ROOT / "results" / "flash_roi_v2_report.csv"
)

MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Save v2 results separately to preserve v1 results
MODEL_PATH = MODEL_DIR / "flash_difference_cnn_v2.pth"
REPORT_PATH = RESULTS_DIR / "flash_cnn_v2_test_report.txt"
USABLE_MANIFEST_PATH = RESULTS_DIR / "flash_cnn_v2_manifest.csv"

SEED = 42
EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 0.0005
IMAGE_SIZE = 128

CLASSES = {
    "untreated": 0,
    "starch_treated": 1,
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# 2. FIND V2 DIFFERENCE IMAGES AND MASKS
# ============================================================

def find_file(folder, sample_id, suffix):
    """
    Find exactly one file matching:
        sample_id + suffix

    Matching is case-insensitive.
    """
    if not folder.exists():
        return None

    matches = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.name.lower() == f"{sample_id}{suffix}".lower()
    ]

    if len(matches) != 1:
        return None

    return matches[0]


def find_difference_image(sample_id, class_name):
    folder = FLASH_DIR / class_name

    return find_file(
        folder,
        sample_id,
        "_difference.jpg",
    )


def find_valid_paneer_mask(sample_id, class_name):
    folder = FLASH_DIR / class_name

    return find_file(
        folder,
        sample_id,
        "_valid_paneer_mask.png",
    )


# ============================================================
# 3. DATASET
# ============================================================

class DifferenceDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]

        path = ROOT / row["difference_path"]

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Cannot read image: {path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        image = image.astype(np.float32) / 255.0

        # Mild horizontal flip augmentation for training only.
        if self.augment and random.random() < 0.5:
            image = np.fliplr(image).copy()

        # HWC -> CHW
        image = torch.from_numpy(image.transpose(2, 0, 1))

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long,
        )

        return image, label


# ============================================================
# 4. CNN MODEL
# ============================================================

class FlashDifferenceCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# 5. LOAD AND VALIDATE DATA
# ============================================================

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {MANIFEST_PATH}"
        )

    if not PIPELINE_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"V2 pipeline report not found: "
            f"{PIPELINE_REPORT_PATH}"
        )

    if not FLASH_DIR.exists():
        raise FileNotFoundError(
            f"V2 output directory not found: {FLASH_DIR}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)
    pipeline_report = pd.read_csv(PIPELINE_REPORT_PATH)

    required_manifest_columns = {
        "sample_id",
        "class_name",
        "label",
        "split",
    }

    missing_columns = required_manifest_columns - set(
        manifest.columns
    )

    if missing_columns:
        raise ValueError(
            "Split manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    required_report_columns = {
        "sample_id",
        "class_name",
        "status",
    }

    missing_columns = required_report_columns - set(
        pipeline_report.columns
    )

    if missing_columns:
        raise ValueError(
            "V2 pipeline report is missing columns: "
            f"{sorted(missing_columns)}"
        )

    # Normalize text fields without changing sample identities.
    manifest["sample_id"] = manifest["sample_id"].astype(str).str.strip()
    manifest["class_name"] = (
        manifest["class_name"].astype(str).str.strip().str.lower()
    )
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()

    pipeline_report["sample_id"] = (
        pipeline_report["sample_id"].astype(str).str.strip()
    )
    pipeline_report["class_name"] = (
        pipeline_report["class_name"].astype(str).str.strip().str.lower()
    )
    pipeline_report["status"] = (
        pipeline_report["status"].astype(str).str.strip().str.lower()
    )

    if manifest["sample_id"].str.lower().duplicated().any():
        raise ValueError(
            "Duplicate sample IDs found in the split manifest. "
            "Inspect the manifest before training."
        )

    allowed_splits = {"train", "validation", "test"}

    if not set(manifest["split"]).issubset(allowed_splits):
        raise ValueError(
            "Unexpected split values found. Expected train, "
            "validation, and test."
        )

    if not set(manifest["class_name"]).issubset(CLASSES):
        raise ValueError(
            "Unexpected class name in the split manifest."
        )

    # Validate labels against class names.
    for _, row in manifest.iterrows():
        expected_label = CLASSES[row["class_name"]]

        if int(row["label"]) != expected_label:
            raise ValueError(
                f"Label mismatch for {row['sample_id']}: "
                f"class={row['class_name']}, "
                f"label={row['label']}"
            )

    # Keep the original sample-level split unchanged.
    # Exclude samples whose v2 pipeline status is not success.
    records = []
    excluded = []

    for _, row in manifest.iterrows():
        sample_id = row["sample_id"]
        class_name = row["class_name"]

        matching_report = pipeline_report[
            (pipeline_report["sample_id"].str.lower()
             == sample_id.lower())
            & (pipeline_report["class_name"] == class_name)
        ]

        if matching_report.empty:
            excluded.append(
                (sample_id, class_name, "Missing from v2 report")
            )
            continue

        if not (matching_report["status"] == "success").all():
            excluded.append(
                (sample_id, class_name, "V2 pipeline failed")
            )
            continue

        difference_path = find_difference_image(
            sample_id,
            class_name,
        )

        if difference_path is None:
            excluded.append(
                (sample_id, class_name, "Difference image missing")
            )
            continue

        mask_path = find_valid_paneer_mask(
            sample_id,
            class_name,
        )

        if mask_path is None:
            excluded.append(
                (sample_id, class_name, "Valid paneer mask missing")
            )
            continue

        # Ensure that the image and mask can both be read.
        image_check = cv2.imread(str(difference_path))
        mask_check = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image_check is None or mask_check is None:
            excluded.append(
                (sample_id, class_name, "Image or mask unreadable")
            )
            continue

        if image_check.shape[:2] != mask_check.shape[:2]:
            excluded.append(
                (sample_id, class_name, "Image/mask size mismatch")
            )
            continue

        records.append({
            "sample_id": sample_id,
            "class_name": class_name,
            "label": int(row["label"]),
            "split": row["split"],
            "difference_path": str(
                difference_path.relative_to(ROOT)
            ),
            "mask_path": str(
                mask_path.relative_to(ROOT)
            ),
        })

    data = pd.DataFrame(records)

    print(f"Device: {DEVICE}")
    print(f"Usable difference images: {len(data)}")

    if excluded:
        print("\nExcluded samples:")
        for sample_id, class_name, reason in excluded:
            print(f"  {class_name}/{sample_id}: {reason}")

    if data.empty:
        raise ValueError(
            "No usable images found. Check the v2 output paths "
            "and pipeline report."
        )

    # The original split must not contain a physical sample
    # in more than one partition.
    if data["sample_id"].str.lower().duplicated().any():
        raise ValueError(
            "A physical sample appears more than once in the "
            "usable dataset. Stop to prevent data leakage."
        )

    print("\nUsable samples by split and class:")
    print(
        pd.crosstab(
            data["split"],
            data["class_name"],
            dropna=False,
        )
    )

    train_df = data[data["split"] == "train"].copy()
    val_df = data[data["split"] == "validation"].copy()
    test_df = data[data["split"] == "test"].copy()

    # Validate partitions before training.
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "At least one split is empty after exclusions. "
            "Inspect the split counts before proceeding."
        )

    if train_df["label"].nunique() != 2:
        raise ValueError(
            "The training set must contain both classes."
        )

    if not set(test_df["label"]).issubset({0, 1}):
        raise ValueError("Unexpected test labels found.")

    print(f"\nTrain samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # Save a record of the actual samples used.
    data.to_csv(USABLE_MANIFEST_PATH, index=False)
    print(f"Saved usable manifest: {USABLE_MANIFEST_PATH}")


    # ========================================================
    # 6. DATA LOADERS
    # ========================================================

    train_loader = DataLoader(
        DifferenceDataset(train_df, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        DifferenceDataset(val_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        DifferenceDataset(test_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )


    # ========================================================
    # 7. INITIALIZE MODEL AND OPTIMIZER
    # ========================================================

    model = FlashDifferenceCNN().to(DEVICE)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["label"].to_numpy(),
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-3,
    )


    # ========================================================
    # 8. TRAIN USING VALIDATION LOSS
    # ========================================================

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_total = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_total += loss.item() * len(labels)

        model.eval()
        val_total = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_total += loss.item() * len(labels)

        train_loss = train_total / len(train_df)
        val_loss = val_total / len(val_df)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train loss={train_loss:.4f} | "
            f"validation loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a checkpoint."
        )

    model.load_state_dict(best_state)

    torch.save(best_state, MODEL_PATH)

    print(f"\nSaved checkpoint: {MODEL_PATH}")
    print(f"Best validation loss: {best_val_loss:.4f}")


    # ========================================================
    # 9. FINAL TEST EVALUATION
    # ========================================================

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))

            predictions = outputs.argmax(dim=1)

            y_pred.extend(predictions.cpu().tolist())
            y_true.extend(labels.tolist())

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["untreated", "starch_treated"],
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    test_accuracy = (
        float(np.mean(np.array(y_true) == np.array(y_pred)))
        if y_true else float("nan")
    )

    print("\nFlash Difference CNN v2 — held-out test set")
    print(f"Test samples: {len(y_true)}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(report)
    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)

    with open(REPORT_PATH, "w") as file:
        file.write("Flash Difference CNN v2 evaluation\n\n")
        file.write(f"Device: {DEVICE}\n")
        file.write(f"Usable samples: {len(data)}\n")
        file.write(f"Train samples: {len(train_df)}\n")
        file.write(f"Validation samples: {len(val_df)}\n")
        file.write(f"Test samples: {len(test_df)}\n")
        file.write(f"Best validation loss: {best_val_loss:.4f}\n")
        file.write(f"Test accuracy: {test_accuracy:.4f}\n\n")
        file.write(report)
        file.write("\nConfusion matrix:\n")
        file.write(str(matrix))
        file.write("\n")

        file.write(
            "\nCaution: this dataset contains very few physical "
            "samples. Test metrics are highly uncertain and "
            "should be treated as preliminary.\n"
        )

    print(f"\nSaved test report: {REPORT_PATH}")
    print(
        "Saved usable manifest: "
        "results/flash_cnn_v2_manifest.csv"
    )


if __name__ == "__main__":
    main()
