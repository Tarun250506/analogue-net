"""Three-branch AnalogueNet fusion training (v2).

Branches:
  1. Ambient image -> frozen StandardScaler + LogisticRegression probabilities
  2. Flash difference image -> frozen pretrained FlashDifferenceCNN features
  3. Farneback optical-flow sequence -> trainable CNN + ConvLSTM encoder

Uses the existing shared split manifest and writes only v2 artifacts.
"""
import copy
import random
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from baseline_model import extract_features
from train_flash_cnn import FlashDifferenceCNN, find_difference_image

SEED = 42
EPOCHS = 30
BATCH_SIZE = 2
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
IMAGE_SIZE = 128
PATIENCE = 8

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "results/shared_split_manifest.csv"
VIDEO_REPORT_PATH = ROOT / "results/video_preprocessing_report.csv"
VIDEO_DIR = ROOT / "results/video_features"
AMBIENT_MODEL_PATH = ROOT / "models/ambient_logreg_v1.joblib"
FLASH_MODEL_PATH = ROOT / "models/flash_difference_cnn_v2.pth"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = MODEL_DIR / "multimodal_fusion_v2.pth"
REPORT_PATH = RESULTS_DIR / "multimodal_fusion_v2_test_report.txt"
PREDICTIONS_PATH = RESULTS_DIR / "multimodal_fusion_v2_predictions.csv"
MANIFEST_OUT = RESULTS_DIR / "multimodal_fusion_v2_manifest.csv"
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


def normalize_sample_id(value):
    value = Path(str(value)).stem.strip()
    for suffix in ("_difference", "_flow", "_press"):
        if value.lower().endswith(suffix):
            value = value[:-len(suffix)]
    return value


def find_column(df, candidates, description):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"Missing {description}; columns: {list(df.columns)}")


def load_manifest():
    for path in (MANIFEST_PATH, VIDEO_REPORT_PATH, AMBIENT_MODEL_PATH, FLASH_MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")
    manifest = pd.read_csv(MANIFEST_PATH)
    video_report = pd.read_csv(VIDEO_REPORT_PATH)
    required = ["sample_id", "class_name", "label", "split", "path"]
    missing = [c for c in required if c not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest is missing columns {missing}. Found: {list(manifest.columns)}")

    id_col = find_column(video_report, ["sample_id"], "video sample ID column")
    status_col = find_column(video_report, ["status"], "video preprocessing status column")
    manifest["sample_id"] = manifest["sample_id"].map(normalize_sample_id)
    manifest["class_name"] = manifest["class_name"].astype(str).str.strip().str.lower()
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()
    manifest["label"] = pd.to_numeric(manifest["label"], errors="raise").astype(int)
    video_report["_id"] = video_report[id_col].map(normalize_sample_id)
    successful = set(video_report.loc[video_report[status_col].astype(str).str.lower().isin(["success", "ok", "true"]), "_id"])

    if manifest["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs found in shared split manifest.")

    rows = []
    for _, row in manifest.iterrows():
        sid, cls, label, split = row["sample_id"], row["class_name"], int(row["label"]), row["split"]
        if cls not in CLASS_NAMES or label != CLASS_NAMES.index(cls):
            raise ValueError(f"Class/label mismatch for {sid}: {cls}, {label}")
        if sid not in successful:
            continue
        ambient_path = ROOT / str(row["path"])
        flash_path = find_difference_image(sid, cls)
        flow_path = VIDEO_DIR / f"{sid}_flow.npy"
        if not ambient_path.exists() or flash_path is None or not Path(flash_path).exists() or not flow_path.exists():
            print(f"Skipping {sid}: missing ambient/flash/flow input")
            continue
        rows.append({"sample_id": sid, "class_name": cls, "label": label, "split": split,
                     "ambient_path": str(ambient_path), "flash_path": str(flash_path), "flow_path": str(flow_path)})
    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError("No samples have all three inputs. Check manifest paths, flash_roi_v2, and video_features.")
    for split in ("train", "validation", "test"):
        if not (data["split"] == split).any():
            raise ValueError(f"No complete three-branch samples in '{split}' split.")
    return data


class ThreeBranchDataset(Dataset):
    def __init__(self, dataframe, ambient_model, augment=False):
        self.data = dataframe.reset_index(drop=True)
        self.ambient_model = ambient_model
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        ambient_features = extract_features(str(row["ambient_path"]))
        if ambient_features is None:
            raise ValueError(f"Could not extract ambient features: {row['ambient_path']}")
        ambient_probs = self.ambient_model.predict_proba(np.asarray(ambient_features, dtype=np.float32).reshape(1, -1))[0].astype(np.float32)

        flash = cv2.imread(str(row["flash_path"]), cv2.IMREAD_COLOR)
        if flash is None:
            raise ValueError(f"Cannot read flash difference image: {row['flash_path']}")
        flash = cv2.cvtColor(flash, cv2.COLOR_BGR2RGB)
        flash = cv2.resize(flash, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        flash = flash.astype(np.float32) / 255.0

        flow = np.load(row["flow_path"]).astype(np.float32)
        if flow.ndim != 4 or flow.shape[-1] != 3 or not np.isfinite(flow).all():
            raise ValueError(f"Invalid flow array for {row['sample_id']}: {flow.shape}")
        if flow.shape[1:3] != (IMAGE_SIZE, IMAGE_SIZE):
            flow = np.stack([cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA) for frame in flow]).astype(np.float32)

        if self.augment and random.random() < 0.5:
            flash = np.flip(flash, axis=1).copy()
            flow = np.flip(flow, axis=2).copy()
            flow[..., 0] *= -1.0

        return {
            "ambient_probs": torch.from_numpy(ambient_probs.copy()),
            "flash": torch.from_numpy(np.ascontiguousarray(flash.transpose(2, 0, 1))),
            "flow": torch.from_numpy(np.ascontiguousarray(flow.transpose(0, 3, 1, 2))),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "sample_id": row["sample_id"],
        }


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class VideoEncoder(nn.Module):
    def __init__(self, hidden_channels=32):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, hidden_channels, 3, stride=2, padding=1), nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True),
        )
        self.hidden_channels = hidden_channels
        self.lstm = ConvLSTMCell(hidden_channels, hidden_channels)

    def forward(self, flow):
        # flow: (B,T,C,H,W); encode each time step and update ConvLSTM state.
        batch, steps, channels, height, width = flow.shape
        h = flow.new_zeros(batch, self.hidden_channels, (height + 3) // 4, (width + 3) // 4)
        c = torch.zeros_like(h)
        for t in range(steps):
            frame_features = self.frame_encoder(flow[:, t])
            h, c = self.lstm(frame_features, (h, c))
        return h.mean(dim=(-2, -1))


class ThreeBranchFusionModel(nn.Module):
    def __init__(self, flash_checkpoint_path):
        super().__init__()
        self.flash_encoder = FlashDifferenceCNN()
        state = torch.load(flash_checkpoint_path, map_location="cpu", weights_only=True)
        self.flash_encoder.load_state_dict(state)
        for parameter in self.flash_encoder.parameters():
            parameter.requires_grad = False
        self.video_encoder = VideoEncoder(hidden_channels=32)
        self.ambient_projection = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32))
        self.flash_projection = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.video_projection = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(32)
        self.classifier = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.25), nn.Linear(32, 2))

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen flash CNN batch-normalization statistics fixed.
        self.flash_encoder.eval()
        return self

    def forward(self, ambient_probs, flash, flow):
        ambient_token = self.ambient_projection(ambient_probs)
        with torch.no_grad():
            flash_features = self.flash_encoder.features(flash).flatten(1)
        flash_token = self.flash_projection(flash_features)
        video_token = self.video_projection(self.video_encoder(flow))
        tokens = torch.stack([ambient_token, flash_token, video_token], dim=1)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        fused = self.norm(tokens + attended).mean(dim=1)
        return self.classifier(fused)


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
    total_loss, all_labels, all_predictions = 0.0, [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            ambient = batch["ambient_probs"].to(DEVICE)
            flash = batch["flash"].to(DEVICE)
            flow = batch["flow"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(ambient, flash, flow)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(logits.argmax(1).detach().cpu().numpy().tolist())
    return total_loss / len(loader.dataset), accuracy_score(all_labels, all_predictions)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ambient_model = joblib.load(AMBIENT_MODEL_PATH)
    data = load_manifest()
    print("\nComplete three-branch samples:", len(data))
    print(data.groupby(["split", "class_name"]).size().unstack(fill_value=0))
    data.to_csv(MANIFEST_OUT, index=False)
    train_df = data[data.split == "train"].copy()
    val_df = data[data.split == "validation"].copy()
    test_df = data[data.split == "test"].copy()
    assert set(train_df.sample_id).isdisjoint(set(val_df.sample_id))
    assert set(train_df.sample_id).isdisjoint(set(test_df.sample_id))
    assert set(val_df.sample_id).isdisjoint(set(test_df.sample_id))

    train_ds = ThreeBranchDataset(train_df, ambient_model, augment=True)
    val_ds = ThreeBranchDataset(val_df, ambient_model)
    test_ds = ThreeBranchDataset(test_df, ambient_model)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ThreeBranchFusionModel(FLASH_MODEL_PATH).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=calculate_class_weights(train_df))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_val_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    print("\nTraining three-branch attention fusion model...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch:02d}/{EPOCHS} | train loss={train_loss:.4f}, acc={train_acc:.3f} | val loss={val_loss:.4f}, acc={val_acc:.3f}")
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES, "image_size": IMAGE_SIZE,
                "best_epoch": best_epoch, "best_validation_loss": best_val_loss,
                "architecture": "Ambient LogisticRegression + FlashDifferenceCNN + Video ConvLSTM + tri-modal self-attention",
                "ambient_model_path": str(AMBIENT_MODEL_PATH.name), "flash_model_path": str(FLASH_MODEL_PATH.name)}, MODEL_PATH)
    print(f"\nSaved checkpoint: {MODEL_PATH}")

    model.eval()
    true_labels, predicted_labels, probabilities, sample_ids = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["ambient_probs"].to(DEVICE), batch["flash"].to(DEVICE), batch["flow"].to(DEVICE))
            probs = torch.softmax(logits, dim=1)
            true_labels.extend(batch["label"].numpy().tolist())
            predicted_labels.extend(probs.argmax(1).cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())
            sample_ids.extend(batch["sample_id"])
    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0)
    matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    report_text = ("AnalogueNet three-branch fusion — held-out test set\n"
                   f"Test samples: {len(true_labels)}\nBest epoch: {best_epoch}\n"
                   f"Best validation loss: {best_val_loss:.4f}\nTest accuracy: {accuracy:.4f}\n\n{report}\n"
                   f"Confusion matrix (rows=true, columns=predicted):\n{matrix}\n\n"
                   "WARNING: Very small sample counts make results exploratory; they do not establish reliable real-world performance.\n")
    print("\n" + report_text)
    REPORT_PATH.write_text(report_text)
    pd.DataFrame({"sample_id": sample_ids, "true_label": true_labels, "predicted_label": predicted_labels,
                  "true_class": [CLASS_NAMES[i] for i in true_labels],
                  "predicted_class": [CLASS_NAMES[i] for i in predicted_labels],
                  "prob_untreated": [p[0] for p in probabilities],
                  "prob_starch_treated": [p[1] for p in probabilities]}).to_csv(PREDICTIONS_PATH, index=False)
    print(f"Saved test report: {REPORT_PATH}\nSaved predictions: {PREDICTIONS_PATH}\nSaved manifest: {MANIFEST_OUT}")


if __name__ == "__main__":
    main()
