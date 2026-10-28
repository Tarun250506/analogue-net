
from pathlib import Path
import cv2
import pandas as pd
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "data_prepared"
REPORT_PATH = ROOT / "results" / "inspection_report.csv"

CLASSES = {
    "untreated": "untreated",
    "starch_treated": "starch_treated",
}

IMAGE_EXTENSIONS = {".heic", ".heif", ".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v"}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for folder_name, label in CLASSES.items():
        source_dir = DATA_DIR / folder_name

        if not source_dir.exists():
            print(f"ERROR: Missing folder: {source_dir}")
            continue

        files = sorted(
            p for p in source_dir.rglob("*")
            if p.is_file() and p.name != ".DS_Store"
        )

        for path in files:
            ext = path.suffix.lower()
            row = {
                "class": label,
                "filename": path.name,
                "sample_id": path.stem.rsplit("_", 1)[0],
                "modality": "unknown",
                "status": "not_processed",
                "details": "",
            }

            stem_lower = path.stem.lower()

            if "_ambient" in stem_lower:
                row["modality"] = "ambient"
            elif "_flash" in stem_lower:
                row["modality"] = "flash"
            elif "_press" in stem_lower:
                row["modality"] = "press"

            try:
                if ext in IMAGE_EXTENSIONS:
                    with Image.open(path) as image:
                        image = image.convert("RGB")
                        output_dir = OUTPUT_DIR / label
                        output_dir.mkdir(parents=True, exist_ok=True)

                        output_path = output_dir / f"{path.stem}.jpg"
                        image.save(output_path, "JPEG", quality=95)

                        row["status"] = "success"
                        row["details"] = (
                            f"{image.width}x{image.height}; "
                            f"saved to {output_path.relative_to(ROOT)}"
                        )

                elif ext in VIDEO_EXTENSIONS:
                    video = cv2.VideoCapture(str(path))
                    if not video.isOpened():
                        raise RuntimeError("OpenCV could not open video")

                    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = video.get(cv2.CAP_PROP_FPS)
                    video.release()

                    row["status"] = "success"
                    row["details"] = (
                        f"{width}x{height}; frames={frames}; fps={fps:.2f}"
                    )
                else:
                    row["status"] = "skipped"
                    row["details"] = "Unsupported file type"

            except Exception as error:
                row["status"] = "error"
                row["details"] = str(error)

            rows.append(row)
            print(
                f"{row['status'].upper():12} "
                f"{label:14} {path.name} "
                f"[{row['modality']}] {row['details']}"
            )

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_PATH, index=False)

    print("\n--- DATASET SUMMARY ---")
    if not report.empty:
        print("\nFiles by class, modality and status:")
        print(
            report.groupby(["class", "modality", "status"])
            .size().to_string()
        )
        print("\nStatus totals:")
        print(report["status"].value_counts().to_string())

    print(f"\nReport saved to: {REPORT_PATH}")
    print(f"Converted images saved under: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
