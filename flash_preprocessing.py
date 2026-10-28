
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_prepared"
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "flash_analysis_v2"

CLASSES = ["untreated", "starch_treated"]


def read_color(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def largest_valid_crop(mask, padding=20):
    """Return a bounding crop around valid warped-image pixels."""
    points = cv2.findNonZero((mask > 0).astype(np.uint8))

    if points is None:
        raise RuntimeError("No valid pixels remain after alignment.")

    x, y, width, height = cv2.boundingRect(points)

    image_height, image_width = mask.shape
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_width, x + width + padding)
    y2 = min(image_height, y + height + padding)

    return x1, y1, x2, y2


def align_and_compare(ambient_path, flash_path, output_prefix):
    ambient = read_color(ambient_path)
    flash = read_color(flash_path)

    ambient_gray = cv2.cvtColor(ambient, cv2.COLOR_BGR2GRAY)
    flash_gray = cv2.cvtColor(flash, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)

    keypoints_a, descriptors_a = orb.detectAndCompute(
        ambient_gray, None
    )
    keypoints_f, descriptors_f = orb.detectAndCompute(
        flash_gray, None
    )

    if descriptors_a is None or descriptors_f is None:
        raise RuntimeError("Could not find enough ORB features.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(descriptors_f, descriptors_a, k=2)

    good_matches = [
        m for pair in pairs
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.75 * n.distance
    ]

    if len(good_matches) < 8:
        raise RuntimeError(
            f"Not enough reliable matches: {len(good_matches)}"
        )

    points_flash = np.float32([
        keypoints_f[m.queryIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)

    points_ambient = np.float32([
        keypoints_a[m.trainIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(
        points_flash,
        points_ambient,
        cv2.RANSAC,
        5.0,
    )

    if homography is None or inlier_mask is None:
        raise RuntimeError("Homography estimation failed.")

    height, width = ambient_gray.shape

    aligned_flash = cv2.warpPerspective(
        flash,
        homography,
        (width, height),
    )

    # Warp an all-white mask to identify pixels that came
    # from the original flash image.
    source_mask = np.full(flash_gray.shape, 255, dtype=np.uint8)

    valid_mask = cv2.warpPerspective(
        source_mask,
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
    )

    valid_mask = (valid_mask > 0).astype(np.uint8) * 255

    difference = cv2.absdiff(ambient, aligned_flash)
    difference_gray = cv2.cvtColor(
        difference, cv2.COLOR_BGR2GRAY
    )

    # Keep invalid warped pixels out of the difference image.
    difference_gray[valid_mask == 0] = 0

    # Crop to the valid image area, not yet to the paneer itself.
    x1, y1, x2, y2 = largest_valid_crop(valid_mask)

    ambient_crop = ambient[y1:y2, x1:x2]
    aligned_crop = aligned_flash[y1:y2, x1:x2]
    difference_crop = difference_gray[y1:y2, x1:x2]
    mask_crop = valid_mask[y1:y2, x1:x2]

    # Black out invalid pixels consistently in the saved outputs.
    aligned_crop = aligned_crop.copy()
    aligned_crop[mask_crop == 0] = 0

    difference_crop = difference_crop.copy()
    difference_crop[mask_crop == 0] = 0

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(output_prefix) + "_ambient_crop.jpg",
        ambient_crop,
    )
    cv2.imwrite(
        str(output_prefix) + "_aligned_flash_crop.jpg",
        aligned_crop,
    )
    cv2.imwrite(
        str(output_prefix) + "_difference_crop.jpg",
        difference_crop,
    )
    cv2.imwrite(
        str(output_prefix) + "_valid_mask.png",
        mask_crop,
    )

    inliers = int(inlier_mask.sum())

    valid_pixels = mask_crop > 0
    mean_difference = (
        float(difference_crop[valid_pixels].mean())
        if np.any(valid_pixels)
        else float("nan")
    )

    return {
        "good_matches": len(good_matches),
        "ransac_inliers": inliers,
        "inlier_ratio": inliers / len(good_matches),
        "valid_pixel_fraction": float(np.mean(valid_mask > 0)),
        "mean_difference_valid_pixels": mean_difference,
        "crop_width": x2 - x1,
        "crop_height": y2 - y1,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for class_name in CLASSES:
        class_dir = DATA_DIR / class_name

        for ambient_path in sorted(class_dir.glob("*_ambient.jpg")):
            sample_id = ambient_path.stem.rsplit("_", 1)[0]
            flash_path = class_dir / f"{sample_id}_flash.jpg"

            row = {
                "class_name": class_name,
                "sample_id": sample_id,
                "status": "error",
                "details": "",
            }

            try:
                if not flash_path.exists():
                    raise FileNotFoundError(
                        f"Missing matching flash image: {flash_path.name}"
                    )

                prefix = OUTPUT_DIR / class_name / sample_id

                metrics = align_and_compare(
                    ambient_path,
                    flash_path,
                    prefix,
                )

                row.update(metrics)
                row["status"] = "success"
                row["details"] = "Aligned and cropped outputs saved."

                print(
                    f"OK {class_name}/{sample_id}: "
                    f"matches={metrics['good_matches']}, "
                    f"inliers={metrics['ransac_inliers']}, "
                    f"valid={metrics['valid_pixel_fraction']:.2f}"
                )

            except Exception as error:
                row["details"] = str(error)
                print(f"ERROR {class_name}/{sample_id}: {error}")

            rows.append(row)

    report_path = RESULTS_DIR / "flash_alignment_report_v2.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False)

    print(f"\nReport saved: {report_path}")
    print(f"Outputs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
