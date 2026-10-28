
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MANIFEST_PATH = ROOT / "results" / "shared_split_manifest.csv"

DATA_DIRS = {
    "untreated": ROOT / "data_prepared" / "untreated",
    "starch_treated": ROOT / "data_prepared" / "starch_treated",
}

OUTPUT_DIR = ROOT / "results" / "video_features"
REPORT_PATH = ROOT / "results" / "video_preprocessing_report.csv"

IMAGE_SIZE = 128
NUM_FRAMES = 16

# Optical flow parameters
FLOW_PYR_SCALE = 0.5
FLOW_LEVELS = 3
FLOW_WINSIZE = 15
FLOW_ITERATIONS = 3
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2


# ============================================================
# VIDEO DISCOVERY
# ============================================================

def find_video(sample_id, class_name):
    """Find the press video for a sample, ignoring case."""

    folder = DATA_DIRS[class_name]

    matches = [
        path for path in folder.iterdir()
        if path.is_file()
        and path.stem.lower() == f"{sample_id}_press".lower()
        and path.suffix.lower() in {".mov", ".mp4", ".avi", ".m4v"}
    ]

    if len(matches) != 1:
        return None

    return matches[0]


# ============================================================
# VIDEO FRAME EXTRACTION
# ============================================================

def extract_frames(video_path):
    """
    Read a video and sample NUM_FRAMES evenly spaced frames
    across its duration.

    Returns RGB frames resized to IMAGE_SIZE x IMAGE_SIZE.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if total_frames < 2:
        cap.release()
        raise ValueError("Video has fewer than two frames.")

    # Avoid sampling duplicate frame indices.
    indices = np.linspace(
        0,
        total_frames - 1,
        num=NUM_FRAMES,
        dtype=int,
    )
    indices = np.unique(indices)

    frames = []

    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = cv2.resize(
            frame,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        frames.append(frame)

    cap.release()

    if len(frames) < 2:
        raise ValueError("Could not extract enough valid frames.")

    # If some frames failed to decode, resample the decoded sequence
    # to a fixed length.
    if len(frames) != NUM_FRAMES:
        positions = np.linspace(
            0,
            len(frames) - 1,
            num=NUM_FRAMES,
        )

        frames = [
            frames[int(round(position))]
            for position in positions
        ]

    return np.asarray(frames, dtype=np.uint8), total_frames, fps


# ============================================================
# FARNEBACK OPTICAL FLOW
# ============================================================

def calculate_optical_flow(frames):
    """
    Compute dense optical flow between consecutive RGB frames.

    Each timestep contains:
        channel 0: horizontal flow (u)
        channel 1: vertical flow (v)
        channel 2: flow magnitude

    Flow components are clipped and scaled for numerical stability.

    Output shape:
        (NUM_FRAMES - 1, IMAGE_SIZE, IMAGE_SIZE, 3)
    """

    flow_sequence = []

    for index in range(len(frames) - 1):
        previous_gray = cv2.cvtColor(
            frames[index],
            cv2.COLOR_RGB2GRAY,
        )

        current_gray = cv2.cvtColor(
            frames[index + 1],
            cv2.COLOR_RGB2GRAY,
        )

        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            FLOW_PYR_SCALE,
            FLOW_LEVELS,
            FLOW_WINSIZE,
            FLOW_ITERATIONS,
            FLOW_POLY_N,
            FLOW_POLY_SIGMA,
            0,
        )

        # Keep the raw flow scale consistent across videos.
        # Clipping reduces the influence of unusually large motion.
        horizontal = np.clip(flow[..., 0], -20.0, 20.0) / 20.0
        vertical = np.clip(flow[..., 1], -20.0, 20.0) / 20.0

        magnitude = np.sqrt(
            flow[..., 0] ** 2 + flow[..., 1] ** 2
        )
        magnitude = np.clip(magnitude, 0.0, 20.0) / 20.0

        flow_features = np.stack(
            [horizontal, vertical, magnitude],
            axis=-1,
        )

        flow_sequence.append(
            flow_features.astype(np.float32)
        )

    return np.asarray(flow_sequence, dtype=np.float32)


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    required_columns = {
        "sample_id",
        "class_name",
        "label",
        "split",
    }

    missing_columns = required_columns - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing_columns)}"
        )

    manifest["sample_id"] = (
        manifest["sample_id"].astype(str).str.strip()
    )
    manifest["class_name"] = (
        manifest["class_name"].astype(str).str.strip().str.lower()
    )
    manifest["split"] = (
        manifest["split"].astype(str).str.strip().str.lower()
    )

    if manifest["sample_id"].str.lower().duplicated().any():
        raise ValueError(
            "Duplicate sample IDs found in the split manifest."
        )

    records = []

    for _, row in manifest.iterrows():
        sample_id = row["sample_id"]
        class_name = row["class_name"]

        if class_name not in DATA_DIRS:
            print(
                f"Skipping {sample_id}: unknown class {class_name}"
            )
            continue

        video_path = find_video(sample_id, class_name)

        if video_path is None:
            print(f"MISSING VIDEO: {class_name}/{sample_id}")
            records.append({
                "sample_id": sample_id,
                "class_name": class_name,
                "label": int(row["label"]),
                "split": row["split"],
                "video_path": "",
                "sequence_path": "",
                "original_frames": 0,
                "fps": 0.0,
                "duration_seconds": 0.0,
                "status": "missing_video",
                "details": "No unique matching press video",
            })
            continue

        try:
            frames, total_frames, fps = extract_frames(video_path)

            flow_sequence = calculate_optical_flow(frames)

            output_path = (
                OUTPUT_DIR / f"{sample_id}_flow.npy"
            )

            # Save optical-flow sequence, not a model checkpoint.
            np.save(output_path, flow_sequence)

            duration = (
                total_frames / fps if fps > 0 else 0.0
            )

            records.append({
                "sample_id": sample_id,
                "class_name": class_name,
                "label": int(row["label"]),
                "split": row["split"],
                "video_path": str(video_path.relative_to(ROOT)),
                "sequence_path": str(output_path.relative_to(ROOT)),
                "original_frames": total_frames,
                "fps": fps,
                "duration_seconds": duration,
                "status": "success",
                "details": "",
            })

            print(
                f"Processed {class_name}/{sample_id} | "
                f"frames={total_frames} | "
                f"flow_shape={flow_sequence.shape}"
            )

        except Exception as error:
            records.append({
                "sample_id": sample_id,
                "class_name": class_name,
                "label": int(row["label"]),
                "split": row["split"],
                "video_path": str(video_path.relative_to(ROOT)),
                "sequence_path": "",
                "original_frames": 0,
                "fps": 0.0,
                "duration_seconds": 0.0,
                "status": "error",
                "details": str(error),
            })

            print(
                f"ERROR {class_name}/{sample_id}: {error}"
            )

    report = pd.DataFrame(records)
    report.to_csv(REPORT_PATH, index=False)

    print("\nPreprocessing summary")
    print("---------------------")
    print(report["status"].value_counts(dropna=False))

    successful = report[report["status"] == "success"]

    if not successful.empty:
        print("\nSuccessful videos by split and class:")
        print(
            pd.crosstab(
                successful["split"],
                successful["class_name"],
            )
        )

    print(f"\nSaved report: {REPORT_PATH}")
    print(f"Saved flow sequences in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
