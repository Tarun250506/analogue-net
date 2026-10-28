
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_prepared"
OUTPUT = ROOT / "results" / "flash_quality_roi_report.csv"


def find_paneer_roi(image):
    """Estimate the paneer bounding box from an ambient image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Paneer is generally brighter than the dark background.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect = w / max(h, 1)

        # Exclude tiny objects and contours spanning nearly the whole image.
        if (
            area > 0.0005 * width * height
            and area < 0.35 * width * height
            and 0.25 <= aspect <= 2.5
        ):
            candidates.append((area, x, y, w, h))

    if not candidates:
        raise ValueError("Could not locate paneer automatically")

    _, x, y, w, h = max(candidates, key=lambda item: item[0])

    # Slight padding keeps the paneer edges in the crop.
    pad_x = int(0.05 * w)
    pad_y = int(0.05 * h)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)

    return x1, y1, x2, y2


def measure_region(image, roi):
    """Measure grayscale brightness and clipping in a region."""
    x1, y1, x2, y2 = roi
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError("Empty paneer crop")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    return {
        "mean_brightness": round(float(gray.mean()), 2),
        "pixels_over_245_pct": round(
            float(np.mean(gray >= 245) * 100), 2
        ),
        "pixels_over_250_pct": round(
            float(np.mean(gray >= 250) * 100), 2
        ),
        "pixels_under_30_pct": round(
            float(np.mean(gray <= 30) * 100), 2
        ),
        "roi_width": int(gray.shape[1]),
        "roi_height": int(gray.shape[0]),
    }


def main():
    rows = []

    for class_name in ["untreated", "starch_treated"]:
        folder = DATA_DIR / class_name

        for flash_path in sorted(folder.glob("*_flash.jpg")):
            # Match the flash photo to its ambient photo by filename.
            ambient_path = folder / f"{flash_path.stem.removesuffix('_flash')}_ambient.jpg"

            if not ambient_path.exists():
                print(f"SKIP {flash_path.name}: ambient pair not found")
                continue

            ambient = cv2.imread(str(ambient_path))
            flash = cv2.imread(str(flash_path))

            if ambient is None or flash is None:
                print(f"SKIP {flash_path.name}: could not read image")
                continue

            try:
                roi = find_paneer_roi(ambient)

                # Use the same coordinates for both images.
                # This assumes the two photos have matching dimensions
                # and the paneer has not moved significantly.
                if ambient.shape[:2] != flash.shape[:2]:
                    print(
                        f"SKIP {flash_path.name}: image dimensions differ; "
                        "align the pair before comparing ROIs"
                    )
                    continue

                ambient_metrics = measure_region(ambient, roi)
                flash_metrics = measure_region(flash, roi)

                rows.append({
                    "sample": flash_path.stem.removesuffix("_flash"),
                    "class": class_name,
                    **{
                        f"ambient_{key}": value
                        for key, value in ambient_metrics.items()
                    },
                    **{
                        f"flash_{key}": value
                        for key, value in flash_metrics.items()
                    },
                    "brightness_change": round(
                        flash_metrics["mean_brightness"]
                        - ambient_metrics["mean_brightness"],
                        2,
                    ),
                })

                print(
                    f"OK {class_name}/{flash_path.stem}: "
                    f"ambient bright pixels="
                    f"{ambient_metrics['pixels_over_250_pct']:.2f}%, "
                    f"flash bright pixels="
                    f"{flash_metrics['pixels_over_250_pct']:.2f}%"
                )

            except ValueError as error:
                print(f"SKIP {flash_path.name}: {error}")

    report = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT, index=False)

    print(f"\nProcessed {len(report)} paired samples.")
    print(f"Saved report to: {OUTPUT}")

    if not report.empty:
        print("\nFlash ROI quality summary:")
        print(
            report[
                [
                    "class",
                    "flash_mean_brightness",
                    "flash_pixels_over_250_pct",
                    "flash_pixels_over_245_pct",
                ]
            ].groupby("class").mean(numeric_only=True).round(2)
        )


if __name__ == "__main__":
    main()
