
from pathlib import Path
import random

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

VIDEO_REPORT = ROOT / "results" / "video_preprocessing_report.csv"
OUTPUT_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "video_convlstm_v1.pth"
REPORT_PATH = OUTPUT_DIR / "video_convlstm_test_report.txt"
PREDICTIONS_PATH = OUTPUT_DIR / "video_convlstm_predictions.csv"

SEED = 42
EPOCHS = 30
BATCH_SIZE = 2
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.001

NUM_FRAMES = 15
IMAGE_SIZE = 128
INPUT_CHANNELS = 3

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
# 2. DATASET
# ============================================================

class OpticalFlowDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]
        path = ROOT / row["sequence_path"]

        sequence = np.load(path).astype(np.float32)

        expected_shape = (
            NUM_FRAMES,
            IMAGE_SIZE,
            IMAGE_SIZE,
            INPUT_CHANNELS,
        )

        if sequence.shape != expected_shape:
            raise ValueError(
                f"Unexpected sequence shape for {path}: "
                f"{sequence.shape}; expected {expected_shape}"
            )

        if not np.isfinite(sequence).all():
            raise ValueError(
                f"Non-finite values found in {path}"
            )

        # Mild horizontal flip, consistently across all timesteps.
        # Horizontal flow changes sign when the image is flipped.
        if self.augment and random.random() < 0.5:
            sequence = np.flip(sequence, axis=2).copy()
            sequence[..., 0] *= -1.0

        # T,H,W,C -> T,C,H,W
        sequence = torch.from_numpy(
            sequence.transpose(0, 3, 1, 2).copy()
        )

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long,
        )

        return sequence, label


# ============================================================
# 3. CONVOLUTIONAL LSTM CELL
# ============================================================

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()

        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Calculate input, forget, output, and candidate gates.
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, hidden_state):
        h, c = hidden_state

        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        h = torch.zeros(
            batch_size,
            self.hidden_channels,
            height,
            width,
            device=device,
        )

        c = torch.zeros_like(h)

        return h, c


# ============================================================
# 4. VIDEO MODEL
# ============================================================

class VideoConvLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        # Encode each optical-flow frame.
        # 128 -> 64 -> 32 -> 16 spatial resolution.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.temporal_model = ConvLSTMCell(
            input_channels=32,
            hidden_channels=32,
            kernel_size=3,
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(32, 2),
        )

    def forward(self, sequence):
        # Input: B,T,C,H,W
        batch_size, timesteps, channels, height, width = sequence.shape

        # Encode every frame using shared convolutional weights.
        frames = sequence.reshape(
            batch_size * timesteps,
            channels,
            height,
            width,
        )

        encoded = self.encoder(frames)

        _, feature_channels, feature_height, feature_width = (
            encoded.shape
        )

        encoded = encoded.reshape(
            batch_size,
            timesteps,
            feature_channels,
            feature_height,
            feature_width,
        )

        state = self.temporal_model.init_hidden(
            batch_size,
            feature_height,
            feature_width,
            sequence.device,
        )

        # Process the sequence in temporal order.
        for timestep in range(timesteps):
            state = self.temporal_model(
                encoded[:, timestep],
                state,
            )

        final_hidden, _ = state

        features = self.pool(final_hidden)

        return self.classifier(features)


# ============================================================
# 5. VALIDATE INPUTS AND LOAD THE SPLITS
# ============================================================

def load_data():
    if not VIDEO_REPORT.exists():
        raise FileNotFoundError(
            f"Video preprocessing report not found: {VIDEO_REPORT}"
        )

    report = pd.read_csv(VIDEO_REPORT)

    required_columns = {
        "sample_id",
        "class_name",
        "label",
        "split",
        "sequence_path",
        "status",
    }

    missing_columns = required_columns - set(report.columns)

    if missing_columns:
        raise ValueError(
            f"Missing report columns: {sorted(missing_columns)}"
        )

    report["sample_id"] = (
        report["sample_id"].astype(str).str.strip()
    )
    report["class_name"] = (
        report["class_name"].astype(str).str.strip().str.lower()
    )
    report["split"] = (
        report["split"].astype(str).str.strip().str.lower()
    )
    report["status"] = (
        report["status"].astype(str).str.strip().str.lower()
    )

    if report["sample_id"].str.lower().duplicated().any():
        raise ValueError(
            "Duplicate sample IDs in video report."
        )

    if not set(report["class_name"]).issubset(CLASSES):
        raise ValueError("Unexpected class names found.")

    if not set(report["split"]).issubset(
        {"train", "validation", "test"}
    ):
        raise ValueError("Unexpected split names found.")

    for _, row in report.iterrows():
        if int(row["label"]) != CLASSES[row["class_name"]]:
            raise ValueError(
                f"Label mismatch for {row['sample_id']}"
            )

    data = report[report["status"] == "success"].copy()

    if len(data) != 20:
        print(
            f"WARNING: Expected 20 successful videos; "
            f"found {len(data)}."
        )

    # Check that all feature files exist.
    valid_paths = []

    for _, row in data.iterrows():
        path = ROOT / str(row["sequence_path"])

        if not path.exists():
            print(f"Missing sequence: {path}")
            valid_paths.append(False)
        else:
            valid_paths.append(True)

    data = data.loc[valid_paths].copy()

    if data.empty:
        raise ValueError("No usable video sequences found.")

    # Each physical sample must appear in only one partition.
    if data["sample_id"].str.lower().duplicated().any():
        raise ValueError(
            "Sample leakage risk: duplicate sample IDs detected."
        )

    train_df = data[data["split"] == "train"].copy()
    val_df = data[data["split"] == "validation"].copy()
    test_df = data[data["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One or more data splits are empty.")

    if train_df["label"].nunique() != 2:
        raise ValueError(
            "Training data must contain both classes."
        )

    print(f"Device: {DEVICE}")
    print(f"Usable video sequences: {len(data)}")

    print("\nSamples by split and class:")
    print(pd.crosstab(data["split"], data["class_name"]))

    print(f"\nTrain: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    return data, train_df, val_df, test_df


# ============================================================
# 6. TRAINING AND VALIDATION
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data, train_df, val_df, test_df = load_data()

    train_loader = DataLoader(
        OpticalFlowDataset(train_df, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        OpticalFlowDataset(val_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        OpticalFlowDataset(test_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = VideoConvLSTM().to(DEVICE)

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
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_state = None

    print("\nStarting ConvLSTM video training...\n")

    for epoch in range(EPOCHS):
        model.train()

        train_total = 0.0

        for sequences, labels in train_loader:
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(sequences)
            loss = criterion(logits, labels)

            loss.backward()

            # Guard against unusually large gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            train_total += loss.item() * len(labels)

        model.eval()

        val_total = 0.0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(sequences)
                loss = criterion(logits, labels)

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
        raise RuntimeError("No model checkpoint was produced.")

    model.load_state_dict(best_state)

    torch.save(
        best_state,
        MODEL_PATH,
    )

    print(f"\nSaved checkpoint: {MODEL_PATH}")
    print(f"Best validation loss: {best_val_loss:.4f}")


    # ========================================================
    # 7. HELD-OUT TEST EVALUATION
    # ========================================================

    model.eval()

    y_true = []
    y_pred = []
    probabilities = []
    sample_ids = []

    # Dataset records preserve the same order as test_df.
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(DEVICE)

            logits = model(sequences)
            probs = torch.softmax(logits, dim=1)

            predictions = probs.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())

    sample_ids = test_df["sample_id"].tolist()

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

    accuracy = float(
        np.mean(np.array(y_true) == np.array(y_pred))
    )

    print("\nConvLSTM video classifier — held-out test set")
    print(f"Test samples: {len(y_true)}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(report)
    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)

    predictions_df = pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": y_true,
        "predicted_label": y_pred,
        "p_untreated": [p[0] for p in probabilities],
        "p_starch_treated": [p[1] for p in probabilities],
    })

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    with open(REPORT_PATH, "w") as file:
        file.write("ConvLSTM video classifier evaluation\n\n")
        file.write(f"Device: {DEVICE}\n")
        file.write(f"Usable samples: {len(data)}\n")
        file.write(f"Train samples: {len(train_df)}\n")
        file.write(f"Validation samples: {len(val_df)}\n")
        file.write(f"Test samples: {len(test_df)}\n")
        file.write(f"Best validation loss: {best_val_loss:.4f}\n")
        file.write(f"Test accuracy: {accuracy:.4f}\n\n")
        file.write(report)
        file.write("\nConfusion matrix:\n")
        file.write(str(matrix))
        file.write(
            "\n\nCaution: only 20 physical samples are available. "
            "Test metrics are exploratory and highly uncertain.\n"
        )

    print(f"\nSaved report: {REPORT_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
