
import copy
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
EPOCHS = 30
BATCH_SIZE = 2
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.001
IMAGE_SIZE = 128
PATIENCE = 8

ROOT = Path(__file__).resolve().parent

MANIFEST_PATH = ROOT / "results/shared_split_manifest.csv"
VIDEO_REPORT_PATH = ROOT / "results/video_preprocessing_report.csv"
IMAGE_DIR = ROOT / "results/flash_roi_v2"
VIDEO_DIR = ROOT / "results/video_features"

MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_PATH = MODEL_DIR / "multimodal_fusion_v1.pth"
REPORT_PATH = RESULTS_DIR / "multimodal_fusion_test_report.txt"
PREDICTIONS_PATH = RESULTS_DIR / "multimodal_fusion_predictions.csv"
MANIFEST_OUT = RESULTS_DIR / "multimodal_fusion_manifest.csv"

CLASS_NAMES = ["untreated", "starch_treated"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")


# ============================================================
# DATASET
# ============================================================

def find_column(df, candidates, description):
    """Find a column without depending on capitalization."""
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Could not find {description}. "
        f"Available columns: {list(df.columns)}"
    )


def normalize_sample_id(value):
    """Standardize IDs while preserving the original sample number."""
    value = Path(str(value)).stem.strip()

    for suffix in ("_difference", "_flow", "_press"):
        if value.lower().endswith(suffix):
            value = value[:-len(suffix)]

    return value


def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")

    if not VIDEO_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Missing video report: {VIDEO_REPORT_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)
    video_report = pd.read_csv(VIDEO_REPORT_PATH)

    for column in ["sample_id", "class_name", "label", "split"]:
        if column not in manifest.columns:
            raise ValueError(
                f"{MANIFEST_PATH.name} must contain '{column}'. "
                f"Found: {list(manifest.columns)}"
            )

    video_id_col = find_column(
        video_report,
        ["sample_id"],
        "video sample ID column"
    )
    status_col = find_column(
        video_report,
        ["status"],
        "video preprocessing status column"
    )

    manifest["sample_id"] = manifest["sample_id"].map(normalize_sample_id)
    manifest["class_name"] = (
        manifest["class_name"].astype(str).str.strip().str.lower()
    )
    manifest["split"] = (
        manifest["split"].astype(str).str.strip().str.lower()
    )
    manifest["label"] = pd.to_numeric(
        manifest["label"], errors="raise"
    ).astype(int)

    video_report["sample_id"] = (
        video_report[video_id_col].map(normalize_sample_id)
    )

    successful = video_report[
        video_report[status_col].astype(str).str.lower().isin(
            ["success", "ok", "true"]
        )
    ]["sample_id"].tolist()

    successful = set(successful)

    if manifest["sample_id"].duplicated().any():
        duplicates = manifest.loc[
            manifest["sample_id"].duplicated(keep=False), "sample_id"
        ].tolist()
        raise ValueError(f"Duplicate sample IDs in manifest: {duplicates}")

    rows = []

    for _, row in manifest.iterrows():
        sample_id = row["sample_id"]
        class_name = row["class_name"]
        label = int(row["label"])
        split = row["split"]

        image_path = (
    ROOT / "results" / "flash_roi_v1" / class_name
    / f"{sample_id}_difference_roi.jpg"
)
        flow_path = VIDEO_DIR / f"{sample_id}_flow.npy"

        # The fusion model requires both modalities.
        if sample_id not in successful:
            print(f"Skipping {sample_id}: video preprocessing unsuccessful")
            continue

        if not image_path.exists():
            print(f"Skipping {sample_id}: missing image {image_path.name}")
            continue

        if not flow_path.exists():
            print(f"Skipping {sample_id}: missing flow {flow_path.name}")
            continue

        if class_name not in CLASS_NAMES:
            raise ValueError(
                f"Unexpected class '{class_name}' for {sample_id}"
            )

        expected_label = CLASS_NAMES.index(class_name)
        if label != expected_label:
            raise ValueError(
                f"Label mismatch for {sample_id}: "
                f"class={class_name}, label={label}"
            )

        rows.append({
            "sample_id": sample_id,
            "class_name": class_name,
            "label": label,
            "split": split,
            "image_path": str(image_path),
            "flow_path": str(flow_path),
        })

    data = pd.DataFrame(rows)

    if data.empty:
        raise ValueError(
            "No paired image/video samples found. "
            "Check IMAGE_DIR, VIDEO_DIR, and sample filenames."
        )

    for split in ["train", "validation", "test"]:
        count = (data["split"] == split).sum()
        if count == 0:
            raise ValueError(f"No paired samples in the '{split}' split.")

    return data


class MultimodalDataset(Dataset):
    def __init__(self, dataframe, augment=False):
        self.data = dataframe.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # Difference image: HWC uint8 -> CHW float32 in [0, 1].
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Cannot read {row['image_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image, (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )
        image = image.astype(np.float32) / 255.0

        # Video optical flow: (T, H, W, C).
        flow = np.load(row["flow_path"]).astype(np.float32)

        if flow.ndim != 4 or flow.shape[-1] != 3:
            raise ValueError(
                f"Expected flow shape (T,H,W,3), got {flow.shape} "
                f"for {row['sample_id']}"
            )

        if not np.isfinite(flow).all():
            raise ValueError(f"Non-finite flow values: {row['sample_id']}")

        # Resize each flow frame if necessary.
        if flow.shape[1:3] != (IMAGE_SIZE, IMAGE_SIZE):
            resized = []
            for frame in flow:
                resized.append(
                    cv2.resize(
                        frame,
                        (IMAGE_SIZE, IMAGE_SIZE),
                        interpolation=cv2.INTER_AREA
                    )
                )
            flow = np.stack(resized).astype(np.float32)

        # Horizontal flipping is applied to both modalities together.
        if self.augment and random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            flow = np.flip(flow, axis=2).copy()
            flow[..., 0] *= -1.0

        image = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        flow = torch.from_numpy(
            np.ascontiguousarray(flow.transpose(0, 3, 1, 2))
        )

        return {
            "image": image,
            "flow": flow,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "sample_id": row["sample_id"],
        }


# ============================================================
# CONVLSTM VIDEO ENCODER
# ============================================================

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x, state):
        h, c = state
        gates = self.gates(torch.cat([x, h], dim=1))

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c

    def init_state(self, batch_size, height, width, device):
        shape = (
            batch_size,
            self.hidden_channels,
            height,
            width
        )
        return (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device)
        )


class VideoEncoder(nn.Module):
    def __init__(self, hidden_channels=32):
        super().__init__()

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.convlstm = ConvLSTMCell(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels
        )

        self.output_channels = hidden_channels

    def forward(self, sequence):
        # sequence: (B, T, 3, H, W)
        batch_size, time_steps, channels, height, width = sequence.shape

        first_frame = self.frame_encoder(sequence[:, 0])
        _, hidden_channels, feat_h, feat_w = first_frame.shape

        state = self.convlstm.init_state(
            batch_size, feat_h, feat_w, sequence.device
        )

        for t in range(time_steps):
            frame_features = self.frame_encoder(sequence[:, t])
            state = self.convlstm(frame_features, state)

        # Final temporal representation: (B, C, H', W').
        return state[0]


# ============================================================
# IMAGE ENCODER
# ============================================================

class ImageEncoder(nn.Module):
    def __init__(self, output_channels=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, output_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.output_channels = output_channels

    def forward(self, image):
        return self.encoder(image)


# ============================================================
# CROSS-ATTENTION FUSION
# ============================================================

class CrossAttentionFusion(nn.Module):
    def __init__(self, channels=32, heads=4):
        super().__init__()

        self.image_to_video = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=0.1,
            batch_first=True
        )

        self.video_to_image = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=0.1,
            batch_first=True
        )

        self.image_norm = nn.LayerNorm(channels)
        self.video_norm = nn.LayerNorm(channels)

        self.fusion = nn.Sequential(
            nn.Linear(channels * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(self, image_features, video_features):
        # Each feature map: (B, C, H, W).
        # Convert spatial locations into tokens: (B, N, C).
        image_tokens = image_features.flatten(2).transpose(1, 2)
        video_tokens = video_features.flatten(2).transpose(1, 2)

        # Image queries video features.
        image_attended, _ = self.image_to_video(
            query=image_tokens,
            key=video_tokens,
            value=video_tokens,
            need_weights=False
        )

        # Video queries image features.
        video_attended, _ = self.video_to_image(
            query=video_tokens,
            key=image_tokens,
            value=image_tokens,
            need_weights=False
        )

        image_tokens = self.image_norm(image_tokens + image_attended)
        video_tokens = self.video_norm(video_tokens + video_attended)

        # Pool attended tokens.
        image_vector = image_tokens.mean(dim=1)
        video_vector = video_tokens.mean(dim=1)

        fused = torch.cat([image_vector, video_vector], dim=1)
        return self.fusion(fused)


class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_encoder = ImageEncoder(output_channels=32)
        self.video_encoder = VideoEncoder(hidden_channels=32)
        self.cross_attention = CrossAttentionFusion(channels=32, heads=4)

    def forward(self, image, flow):
        image_features = self.image_encoder(image)
        video_features = self.video_encoder(flow)

        return self.cross_attention(image_features, video_features)


# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def calculate_class_weights(train_df):
    counts = train_df["label"].value_counts().to_dict()

    weights = []
    for label in range(2):
        if counts.get(label, 0) == 0:
            raise ValueError(f"Training split has no samples for label {label}.")
        weights.append(len(train_df) / (2.0 * counts[label]))

    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in loader:
            images = batch["image"].to(DEVICE)
            flows = batch["flow"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images, flows)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)

            predictions = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)

    return average_loss, accuracy


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_manifest()

    print("\nPaired multimodal samples:")
    print(f"Total: {len(data)}")
    print(data.groupby(["split", "class_name"]).size().unstack(fill_value=0))

    data.to_csv(MANIFEST_OUT, index=False)

    train_df = data[data["split"] == "train"].copy()
    val_df = data[data["split"] == "validation"].copy()
    test_df = data[data["split"] == "test"].copy()

    # Guard against leakage between splits.
    train_ids = set(train_df["sample_id"])
    val_ids = set(val_df["sample_id"])
    test_ids = set(test_df["sample_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    train_dataset = MultimodalDataset(train_df, augment=True)
    val_dataset = MultimodalDataset(val_df, augment=False)
    test_dataset = MultimodalDataset(test_df, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = MultimodalModel().to(DEVICE)

    class_weights = calculate_class_weights(train_df)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    print("\nTraining multimodal cross-attention model...")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={train_loss:.4f}, acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f}, acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "architecture": "ImageEncoder + ConvLSTM + bidirectional cross-attention",
        },
        MODEL_PATH
    )

    print(f"\nSaved checkpoint: {MODEL_PATH}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Held-out test evaluation.
    model.eval()
    true_labels = []
    predicted_labels = []
    probabilities = []
    sample_ids = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(DEVICE)
            flows = batch["flow"].to(DEVICE)
            labels = batch["label"]

            logits = model(images, flows)
            probs = torch.softmax(logits, dim=1)
            predictions = probs.argmax(dim=1).cpu().numpy()

            true_labels.extend(labels.numpy().tolist())
            predicted_labels.extend(predictions.tolist())
            probabilities.extend(probs.cpu().numpy().tolist())
            sample_ids.extend(batch["sample_id"])

    accuracy = accuracy_score(true_labels, predicted_labels)

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0
    )

    matrix = confusion_matrix(
        true_labels, predicted_labels, labels=[0, 1]
    )

    report_text = (
        "Multimodal cross-attention model — held-out test set\n"
        f"Test samples: {len(true_labels)}\n"
        f"Best epoch: {best_epoch}\n"
        f"Best validation loss: {best_val_loss:.4f}\n"
        f"Test accuracy: {accuracy:.4f}\n\n"
        f"{report}\n"
        "Confusion matrix (rows=true, columns=predicted):\n"
        f"{matrix}\n\n"
        "WARNING: This dataset contains very few independent physical samples. "
        "Results are exploratory and do not establish reliable real-world "
        "classification performance.\n"
    )

    print("\n" + report_text)

    REPORT_PATH.write_text(report_text)

    predictions_df = pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": true_labels,
        "predicted_label": predicted_labels,
        "true_class": [CLASS_NAMES[i] for i in true_labels],
        "predicted_class": [CLASS_NAMES[i] for i in predicted_labels],
        "prob_untreated": [p[0] for p in probabilities],
        "prob_starch_treated": [p[1] for p in probabilities],
    })

    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Saved test report: {REPORT_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")
    print(f"Saved paired manifest: {MANIFEST_OUT}")


if __name__ == "__main__":
    main()
