
from pathlib import Path
import traceback

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_prepared"

SEGMENTATION_DIR = ROOT / "results" / "segmentation_qc_v2"
OUTPUT_DIR = ROOT / "results" / "flash_roi_v2"
REPORT_PATH = ROOT / "results" / "flash_roi_v2_report.csv"

CLASSES = ["untreated", "starch_treated"]

# Alignment quality thresholds
MIN_MATCHES = 8
MIN_INLIER_RATIO = 0.25
MIN_VALID_PANEER_FRACTION = 0.80

# Small erosion reduces the effect of uncertain mask boundaries.
# Set to 0 to use the complete segmentation mask.
MASK_EROSION_PIXELS = 2


def read_image(path):
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    return image


def load_paneer_mask(sample_id, class_name, image_shape):
    """
    Load the existing segmentation mask.

    Expected path:
    results/segmentation_qc_v2/<class>/<sample>_ambient_mask.png

    Returns a binary mask with values 0 and 255.
    """
    mask_path = (
        SEGMENTATION_DIR
        / class_name
        / f"{sample_id}_ambient_mask.png"
    )

    # Fall back to case-insensitive filename matching.
    if not mask_path.exists():
        folder = SEGMENTATION_DIR / class_name

        candidates = [
            path
            for path in folder.glob("*_ambient_mask.png")
            if path.stem.rsplit("_ambient_mask", 1)[0].lower()
            == sample_id.lower()
        ]

        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Could not uniquely find segmentation mask for "
                f"{class_name}/{sample_id} in {folder}"
            )

        mask_path = candidates[0]

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Could not read segmentation mask: {mask_path}")

    height, width = image_shape[:2]

    if mask.shape != (height, width):
        raise ValueError(
            f"Mask/image dimensions differ for {sample_id}: "
            f"mask={mask.shape}, image={(height, width)}"
        )

    binary_mask = (mask > 0).astype(np.uint8) * 255

    if cv2.countNonZero(binary_mask) == 0:
        raise RuntimeError(f"Segmentation mask is empty: {mask_path}")

    if MASK_EROSION_PIXELS > 0:
        kernel_size = 2 * MASK_EROSION_PIXELS + 1
        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )

        eroded = cv2.erode(binary_mask, kernel, iterations=1)

        # Do not silently replace a valid mask with an empty one.
        if cv2.countNonZero(eroded) > 0:
            binary_mask = eroded

    return binary_mask, mask_path


def align_flash_to_ambient(ambient, flash):
    """
    Align flash to ambient with ORB features and RANSAC.

    The homography maps flash-image coordinates into ambient-image
    coordinates.
    """
    if ambient is None or flash is None:
        raise ValueError("Ambient or flash image is missing.")

    height, width = ambient.shape[:2]

    if flash.shape[:2] != (height, width):
        flash = cv2.resize(
            flash,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )

    ambient_gray = cv2.cvtColor(ambient, cv2.COLOR_BGR2GRAY)
    flash_gray = cv2.cvtColor(flash, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=3000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    best_result = None

    # Use the full image for registration. The paneer mask is applied
    # after alignment so that registration can also use background
    # texture where helpful.
    kp_a, desc_a = orb.detectAndCompute(ambient_gray, None)
    kp_f, desc_f = orb.detectAndCompute(flash_gray, None)

    if desc_a is None or desc_f is None:
        raise RuntimeError("Could not extract ORB descriptors.")

    pairs = matcher.knnMatch(desc_f, desc_a, k=2)

    good = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]

    if len(good) < MIN_MATCHES:
        raise RuntimeError(
            f"Too few feature matches: {len(good)} "
            f"(minimum {MIN_MATCHES})"
        )

    source = np.float32([
        kp_f[match.queryIdx].pt for match in good
    ]).reshape(-1, 1, 2)

    destination = np.float32([
        kp_a[match.trainIdx].pt for match in good
    ]).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(
        source,
        destination,
        cv2.RANSAC,
        5.0,
    )

    if homography is None or inlier_mask is None:
        raise RuntimeError("RANSAC could not estimate a homography.")

    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / len(good)

    if inlier_ratio < MIN_INLIER_RATIO:
        raise RuntimeError(
            f"Unreliable alignment: inlier ratio={inlier_ratio:.2f}, "
            f"minimum={MIN_INLIER_RATIO:.2f}"
        )

    aligned = cv2.warpPerspective(
        flash,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Track which pixels originate from the flash image rather than
    # from the black border introduced by perspective warping.
    source_valid = np.full(
        flash.shape[:2],
        255,
        dtype=np.uint8,
    )

    valid_mask = cv2.warpPerspective(
        source_valid,
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    valid_mask = (valid_mask > 0).astype(np.uint8)

    return aligned, valid_mask, len(good), inliers, inlier_ratio


def process_pair(ambient_path, flash_path, class_name, output_dir):
    ambient = read_image(ambient_path)
    flash = read_image(flash_path)

    sample_id = ambient_path.stem.rsplit("_ambient", 1)[0]

    paneer_mask, mask_path = load_paneer_mask(
        sample_id,
        class_name,
        ambient.shape,
    )

    (
        aligned_flash,
        alignment_valid,
        matches,
        inliers,
        inlier_ratio,
    ) = align_flash_to_ambient(ambient, flash)

    # The paneer mask is defined in ambient-image coordinates.
    # Intersect it with the valid warped-flash region.
    paneer_pixels = paneer_mask > 0
    valid_pixels = alignment_valid > 0
    valid_paneer_pixels = paneer_pixels & valid_pixels

    paneer_pixel_count = int(paneer_pixels.sum())
    valid_paneer_count = int(valid_paneer_pixels.sum())

    if paneer_pixel_count == 0 or valid_paneer_count == 0:
        raise RuntimeError(
            "No valid overlapping pixels inside the paneer mask."
        )

    valid_paneer_fraction = (
        valid_paneer_count / paneer_pixel_count
    )

    if valid_paneer_fraction < MIN_VALID_PANEER_FRACTION:
        raise RuntimeError(
            "Insufficient flash/ambient overlap within paneer: "
            f"{valid_paneer_fraction:.3f}"
        )

    ambient_gray = cv2.cvtColor(ambient, cv2.COLOR_BGR2GRAY)
    flash_gray = cv2.cvtColor(aligned_flash, cv2.COLOR_BGR2GRAY)

    # Difference image is zero everywhere outside the valid paneer.
    difference = cv2.absdiff(ambient_gray, flash_gray)
    difference[~valid_paneer_pixels] = 0

    ambient_masked = ambient.copy()
    ambient_masked[~paneer_pixels] = 0

    flash_masked = aligned_flash.copy()
    flash_masked[~valid_paneer_pixels] = 0

    # Compute statistics exclusively over valid paneer pixels.
    ambient_values = ambient_gray[valid_paneer_pixels]
    flash_values = flash_gray[valid_paneer_pixels]
    difference_values = difference[valid_paneer_pixels]

    mean_difference = float(difference_values.mean())
    std_difference = float(difference_values.std())
    p95_difference = float(np.percentile(difference_values, 95))
    max_difference = int(difference_values.max())

    ambient_bright_pct = float(
        np.mean(ambient_values >= 250) * 100
    )
    flash_bright_pct = float(
        np.mean(flash_values >= 250) * 100
    )

    ambient_near_white_pct = float(
        np.mean(ambient_values >= 245) * 100
    )
    flash_near_white_pct = float(
        np.mean(flash_values >= 245) * 100
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save original-size visual checks.
    overlay = ambient.copy()
    contours, _ = cv2.findContours(
        paneer_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 5)

    cv2.imwrite(
        str(output_dir / f"{sample_id}_ambient_overlay.jpg"),
        overlay,
    )
    cv2.imwrite(
        str(output_dir / f"{sample_id}_ambient_masked.jpg"),
        ambient_masked,
    )
    cv2.imwrite(
        str(output_dir / f"{sample_id}_aligned_flash_masked.jpg"),
        flash_masked,
    )
    cv2.imwrite(
        str(output_dir / f"{sample_id}_difference.jpg"),
        difference,
    )
    cv2.imwrite(
        str(output_dir / f"{sample_id}_paneer_mask.png"),
        paneer_mask,
    )
    cv2.imwrite(
        str(output_dir / f"{sample_id}_valid_paneer_mask.png"),
        valid_paneer_pixels.astype(np.uint8) * 255,
    )

    print(
        f"OK {class_name}/{sample_id}: "
        f"matches={matches}, inlier_ratio={inlier_ratio:.2f}, "
        f"valid_paneer={valid_paneer_fraction:.3f}, "
        f"mean_diff={mean_difference:.2f}"
    )

    return {
        "sample_id": sample_id,
        "class_name": class_name,
        "ambient_path": str(ambient_path.relative_to(ROOT)),
        "flash_path": str(flash_path.relative_to(ROOT)),
        "mask_path": str(mask_path.relative_to(ROOT)),
        "good_matches": matches,
        "ransac_inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "paneer_pixels": paneer_pixel_count,
        "valid_paneer_pixels": valid_paneer_count,
        "valid_paneer_fraction": valid_paneer_fraction,
        "mean_difference": mean_difference,
        "std_difference": std_difference,
        "p95_difference": p95_difference,
        "max_difference": max_difference,
        "ambient_pixels_over_250_pct": ambient_bright_pct,
        "flash_pixels_over_250_pct": flash_bright_pct,
        "ambient_pixels_over_245_pct": ambient_near_white_pct,
        "flash_pixels_over_245_pct": flash_near_white_pct,
        "status": "success",
        "details": "",
    }


def main():
    rows = []

    for class_name in CLASSES:
        class_dir = DATA_DIR / class_name
        output_dir = OUTPUT_DIR / class_name

        if not class_dir.exists():
            print(f"WARNING: missing data directory: {class_dir}")
            continue

        ambient_paths = sorted(class_dir.glob("*_ambient.jpg"))

        for ambient_path in ambient_paths:
            sample_id = ambient_path.stem.rsplit("_ambient", 1)[0]
            flash_path = class_dir / f"{sample_id}_flash.jpg"

            try:
                if not flash_path.exists():
                    # Handle sample filename capitalization differences.
                    matches = [
                        path
                        for path in class_dir.glob("*_flash.jpg")
                        if path.stem.rsplit("_flash", 1)[0].lower()
                        == sample_id.lower()
                    ]

                    if len(matches) != 1:
                        raise FileNotFoundError(
                            f"Could not find a unique flash image "
                            f"for {sample_id}"
                        )

                    flash_path = matches[0]

                row = process_pair(
                    ambient_path,
                    flash_path,
                    class_name,
                    output_dir,
                )

            except Exception as error:
                print(f"ERROR {class_name}/{sample_id}: {error}")
                traceback.print_exc()

                row = {
                    "sample_id": sample_id,
                    "class_name": class_name,
                    "status": "error",
                    "details": str(error),
                }

            rows.append(row)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORT_PATH, index=False)

    successes = sum(row["status"] == "success" for row in rows)
    failures = sum(row["status"] == "error" for row in rows)

    print("\nFlash ROI v2 complete")
    print(f"Processed: {len(rows)}")
    print(f"Passed: {successes}")
    print(f"Failed: {failures}")
    print(f"Report: {REPORT_PATH}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
