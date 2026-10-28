from pathlib import Path
import argparse

import cv2
import joblib
import numpy as np
import torch

from baseline_model import extract_features
from train_multimodal_fusion_v2 import (
    ThreeBranchFusionModel,
    DEVICE,
    IMAGE_SIZE,
    CLASS_NAMES,
    AMBIENT_MODEL_PATH,
    FLASH_MODEL_PATH,
)

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "multimodal_fusion_v2.pth"


def preprocess_difference_image(image_path):
    """Preprocess the v2 flash-difference image as during training."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read flash difference image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
    )
    image = image.astype(np.float32) / 255.0
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0)


def preprocess_press_video(video_path, num_frames=16):
    """Convert a pressing video into the normalized Farneback flow sequence."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open pressing video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 2:
        cap.release()
        raise ValueError("Video must contain at least two frames.")

    indices = np.linspace(0, frame_count - 1, num_frames).round().astype(int)
    frames = []
    current_index = 0
    target_position = 0

    while target_position < len(indices):
        ok, frame = cap.read()
        if not ok:
            break

        if current_index == indices[target_position]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(
                frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
            )
            frames.append(frame)
            target_position += 1

        current_index += 1

    cap.release()

    if len(frames) != num_frames:
        raise ValueError(
            f"Expected {num_frames} sampled frames; got {len(frames)}."
        )

    gray_frames = [
        cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames
    ]
    flow_sequence = []

    for previous, current in zip(gray_frames[:-1], gray_frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            previous, current, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flow_x = np.clip(flow[..., 0], -20, 20) / 20.0
        flow_y = np.clip(flow[..., 1], -20, 20) / 20.0
        magnitude = np.clip(magnitude, 0, 20) / 20.0
        flow_sequence.append(np.stack([flow_x, flow_y, magnitude], axis=-1))

    flow_sequence = np.asarray(flow_sequence, dtype=np.float32)
    # (T,H,W,C) -> (1,T,C,H,W)
    flow_tensor = torch.from_numpy(
        np.ascontiguousarray(flow_sequence.transpose(0, 3, 1, 2))
    ).unsqueeze(0)
    return flow_tensor


def load_models():
    """Load the ambient classifier, v2 flash CNN and v2 fusion checkpoint."""
    for path in (CHECKPOINT, AMBIENT_MODEL_PATH, FLASH_MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required model file not found: {path}")

    ambient_model = joblib.load(AMBIENT_MODEL_PATH)
    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=True
    )

    model = ThreeBranchFusionModel(FLASH_MODEL_PATH).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return ambient_model, model


def predict(ambient_image, difference_image, press_video):
    """Predict using ambient, flash-difference and video inputs."""
    ambient_image = Path(ambient_image)
    difference_image = Path(difference_image)
    press_video = Path(press_video)

    for path in (ambient_image, difference_image, press_video):
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

    ambient_model, fusion_model = load_models()

    features = extract_features(str(ambient_image))
    if features is None:
        raise ValueError(f"Could not extract ambient features: {ambient_image}")

    ambient_probs = ambient_model.predict_proba(
        np.asarray(features, dtype=np.float32).reshape(1, -1)
    )[0].astype(np.float32)
    ambient_tensor = torch.from_numpy(ambient_probs.copy()).unsqueeze(0).to(DEVICE)

    flash_tensor = preprocess_difference_image(difference_image).to(DEVICE)
    flow_tensor = preprocess_press_video(press_video).to(DEVICE)

    with torch.no_grad():
        logits = fusion_model(ambient_tensor, flash_tensor, flow_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    predicted_index = int(np.argmax(probabilities))
    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "untreated_score": float(probabilities[0]),
        "starch_treated_score": float(probabilities[1]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="AnalogueNet v2 three-branch experimental inference"
    )
    parser.add_argument("--ambient-image", required=True,
                        help="Original ambient image for the same sample")
    parser.add_argument("--difference-image", required=True,
                        help="Flash-difference image produced by flash_roi_v2")
    parser.add_argument("--video", required=True,
                        help="Pressing video for the same sample")
    args = parser.parse_args()

    result = predict(args.ambient_image, args.difference_image, args.video)

    print("\n--- AnalogueNet v2 three-branch prediction ---")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Untreated score: {result['untreated_score']:.4f}")
    print(f"Starch-treated score: {result['starch_treated_score']:.4f}")
    print("\nResearch prototype only; not a validated food-safety determination.")


if __name__ == "__main__":
    main()
