
from pathlib import Path
import argparse

import cv2
import numpy as np
import torch

from train_multimodal_fusion import (
    MultimodalModel,
    DEVICE,
    IMAGE_SIZE,
    CLASS_NAMES,
)


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "multimodal_fusion_v1.pth"


def preprocess_difference_image(image_path):
    """Prepare a difference image in the same format as training."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    image = image.astype(np.float32) / 255.0

    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0)


def preprocess_press_video(video_path, num_frames=16):
    """
    Convert a press video into a Farneback optical-flow sequence.

    Keep this preprocessing identical to preprocess_video.py.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count < 2:
        cap.release()
        raise ValueError("Video must contain at least two frames.")

    indices = np.linspace(
        0, frame_count - 1, num_frames
    ).round().astype(int)

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
                frame,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_AREA,
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
        cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        for frame in frames
    ]

    flow_sequence = []

    for previous, current in zip(
        gray_frames[:-1], gray_frames[1:]
    ):
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        magnitude = np.sqrt(
            flow[..., 0] ** 2 + flow[..., 1] ** 2
        )

        # Match the flow normalization used in training.
        flow_x = np.clip(flow[..., 0], -20, 20) / 20.0
        flow_y = np.clip(flow[..., 1], -20, 20) / 20.0
        magnitude = np.clip(magnitude, 0, 20) / 20.0

        flow_sequence.append(
            np.stack([flow_x, flow_y, magnitude], axis=-1)
        )

    flow_sequence = np.asarray(flow_sequence, dtype=np.float32)

    # (T, H, W, C) -> (1, T, C, H, W)
    flow_tensor = torch.from_numpy(
        np.ascontiguousarray(
            flow_sequence.transpose(0, 3, 1, 2)
        )
    ).unsqueeze(0)

    return flow_tensor


def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model = MultimodalModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def predict(difference_image, press_video):
    model = load_model()

    image_tensor = preprocess_difference_image(
        difference_image
    ).to(DEVICE)

    flow_tensor = preprocess_press_video(
        press_video
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor, flow_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    predicted_index = int(np.argmax(probabilities))

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "untreated_score": float(probabilities[0]),
        "starch_treated_score": float(probabilities[1]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Experimental AnalogueNet inference"
    )
    parser.add_argument(
        "--difference-image",
        required=True,
        help="Preprocessed flash/ambient difference image",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Pressing video for the same paneer sample",
    )

    args = parser.parse_args()

    result = predict(args.difference_image, args.video)

    print("\n--- AnalogueNet experimental prediction ---")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Untreated score: {result['untreated_score']:.4f}")
    print(
        "Starch-treated score: "
        f"{result['starch_treated_score']:.4f}"
    )
    print(
        "\nResearch prototype only: this result is not a "
        "validated food-safety or quality determination."
    )


if __name__ == "__main__":
    main()
