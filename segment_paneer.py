
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_prepared"
OUTPUT_DIR = ROOT / "results" / "segmentation_qc_v2"

CLASSES = ["untreated", "starch_treated"]

# Heuristic review threshold, not a calibrated probability.
MIN_AREA_FRACTION = 0.001
MAX_AREA_FRACTION = 0.20
MIN_SCORE = 0.25

PADDING = 10
THUMBNAIL_SIZE = (320, 240)


def make_candidates(gray):
    """Generate foreground candidates using several threshold methods."""
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    candidates = []

    _, otsu = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    candidates.append(("otsu", otsu))

    block_size = 51 if min(gray.shape) > 51 else 11
    if block_size % 2 == 0:
        block_size += 1

    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        5,
    )
    candidates.append(("adaptive", adaptive))
    candidates.append(("otsu_inverse", cv2.bitwise_not(otsu)))

    cleaned = []

    for method, mask in candidates:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )
        cleaned.append((method, mask))

    return cleaned


def score_contour(contour, gray, image_area):
    """Score a plausible object contour using shape and brightness."""
    area = cv2.contourArea(contour)
    area_fraction = area / image_area

    if not MIN_AREA_FRACTION <= area_fraction <= MAX_AREA_FRACTION:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    height, width = gray.shape

    if w <= 0 or h <= 0:
        return None

    # Reject contours that touch the image boundary.
    if (
        x <= 2
        or y <= 2
        or x + w >= width - 2
        or y + h >= height - 2
    ):
        return None

    aspect_ratio = w / h
    if not 0.25 <= aspect_ratio <= 4.0:
        return None

    rectangularity = area / float(w * h)

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    circularity = min(
        4.0 * np.pi * area / (perimeter * perimeter),
        1.0,
    )

    local_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(
        local_mask,
        [contour],
        -1,
        255,
        cv2.FILLED,
    )

    inner_pixels = gray[local_mask > 0]
    if inner_pixels.size == 0:
        return None

    dilated = cv2.dilate(
        local_mask,
        np.ones((31, 31), np.uint8),
    )
    ring = (dilated > 0) & (local_mask == 0)
    outer_pixels = gray[ring]

    inside_brightness = float(inner_pixels.mean())
    outside_brightness = (
        float(outer_pixels.mean())
        if outer_pixels.size
        else inside_brightness
    )

    brightness_contrast = inside_brightness - outside_brightness

    shape_score = np.clip(rectangularity, 0.0, 1.0)
    roundness_score = np.clip(circularity, 0.0, 1.0)
    contrast_score = np.clip(
        (brightness_contrast + 40.0) / 100.0,
        0.0,
        1.0,
    )

    score = (
        0.45 * shape_score
        + 0.20 * roundness_score
        + 0.35 * contrast_score
    )

    return {
        "score": float(score),
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "area_fraction": float(area_fraction),
        "rectangularity": float(rectangularity),
        "circularity": float(circularity),
        "brightness_contrast": float(brightness_contrast),
    }


def segment_paneer(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    image_area = height * width

    candidates = []

    for method, threshold in make_candidates(gray):
        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            metrics = score_contour(contour, gray, image_area)

            if metrics is not None:
                candidates.append({
                    **metrics,
                    "method": method,
                    "contour": contour,
                })

    if not candidates:
        raise RuntimeError(
            "No plausible paneer candidate found. Manual review required."
        )

    best = max(candidates, key=lambda item: item["score"])
    contour = best["contour"]

    x, y, w, h = (
        best["x"],
        best["y"],
        best["w"],
        best["h"],
    )

    # ----------------------------------------------------------
    # FULL-IMAGE MASK FIX
    # Create the mask in the ORIGINAL image coordinate system.
    # Do not save the cropped mask as the main segmentation mask.
    # ----------------------------------------------------------
    full_mask = np.zeros((height, width), dtype=np.uint8)

    cv2.drawContours(
        full_mask,
        [contour],
        -1,
        255,
        cv2.FILLED,
    )

    # Erode slightly to reduce uncertain boundary pixels.
    full_mask = cv2.erode(
        full_mask,
        np.ones((5, 5), np.uint8),
        iterations=1,
    )

    # Compute a padded crop for visual inspection only.
    x1 = max(0, x - PADDING)
    y1 = max(0, y - PADDING)
    x2 = min(width, x + w + PADDING)
    y2 = min(height, y + h + PADDING)

    crop = image[y1:y2, x1:x2]
    crop_mask = full_mask[y1:y2, x1:x2]

    masked_crop = cv2.bitwise_and(
        crop,
        crop,
        mask=crop_mask,
    )

    # Draw the selected contour on the original image.
    overlay = image.copy()
    cv2.drawContours(
        overlay,
        [contour],
        -1,
        (0, 255, 0),
        5,
    )

    cv2.putText(
        overlay,
        f'{best["method"]} score={best["score"]:.2f}',
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    output_dir = OUTPUT_DIR / image_path.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    # Main mask: same dimensions as the original ambient image.
    cv2.imwrite(
        str(output_dir / f"{stem}_mask.png"),
        full_mask,
    )

    # Save the remaining diagnostic outputs.
    cv2.imwrite(
        str(output_dir / f"{stem}_crop.jpg"),
        crop,
    )
    cv2.imwrite(
        str(output_dir / f"{stem}_crop_mask.png"),
        crop_mask,
    )
    cv2.imwrite(
        str(output_dir / f"{stem}_masked.jpg"),
        masked_crop,
    )
    cv2.imwrite(
        str(output_dir / f"{stem}_overlay.jpg"),
        overlay,
    )

    flags = []
    if best["score"] < MIN_SCORE:
        flags.append("low_candidate_score")

    return {
        "sample_id": stem.removesuffix("_ambient"),
        "class_name": image_path.parent.name,
        "image": str(image_path.relative_to(ROOT)),
        "method": best["method"],
        "score": best["score"],
        "bbox_x": x,
        "bbox_y": y,
        "bbox_width": w,
        "bbox_height": h,
        "object_area_fraction": best["area_fraction"],
        "rectangularity": best["rectangularity"],
        "circularity": best["circularity"],
        "brightness_contrast": best["brightness_contrast"],
        "mask_fraction": float(np.mean(full_mask > 0)),
        "mask_height": height,
        "mask_width": width,
        "status": "REVIEW" if flags else "CHECK_VISUALLY",
        "flags": ";".join(flags),
        "overlay_path": str(
            (output_dir / f"{stem}_overlay.jpg").relative_to(ROOT)
        ),
    }


def create_contact_sheet(records):
    tiles = []

    for record in records:
        image_path = ROOT / record["overlay_path"]
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        image = cv2.resize(
            image,
            THUMBNAIL_SIZE,
            interpolation=cv2.INTER_AREA,
        )

        cv2.rectangle(
            image,
            (0, 0),
            (image.shape[1], 52),
            (0, 0, 0),
            cv2.FILLED,
        )

        label = f'{record["class_name"]}/{record["sample_id"]}'

        cv2.putText(
            image,
            label[:42],
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f'{record["status"]} score={record["score"]:.2f}',
            (6, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 200, 255)
            if record["status"] == "REVIEW"
            else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        tiles.append(image)

    if not tiles:
        print("No contact-sheet images available.")
        return

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    tile_h, tile_w = tiles[0].shape[:2]

    sheet = np.zeros(
        (rows * tile_h, cols * tile_w, 3),
        dtype=np.uint8,
    )

    for index, tile in enumerate(tiles):
        r, c = divmod(index, cols)
        sheet[
            r * tile_h:(r + 1) * tile_h,
            c * tile_w:(c + 1) * tile_w,
        ] = tile

    contact_path = OUTPUT_DIR / "segmentation_contact_sheet.jpg"
    cv2.imwrite(str(contact_path), sheet)

    print(f"Contact sheet saved: {contact_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    errors = []

    for class_name in CLASSES:
        class_dir = DATA_DIR / class_name

        if not class_dir.exists():
            print(f"Missing directory: {class_dir}")
            continue

        for image_path in sorted(class_dir.glob("*_ambient.jpg")):
            try:
                record = segment_paneer(image_path)
                records.append(record)

                print(
                    f'{record["status"]}: {class_name}/'
                    f'{image_path.name} | '
                    f'method={record["method"]} | '
                    f'score={record["score"]:.3f} | '
                    f'mask={record["mask_width"]}x'
                    f'{record["mask_height"]}'
                )

            except Exception as error:
                errors.append({
                    "image": str(image_path.relative_to(ROOT)),
                    "error": str(error),
                })
                print(f"ERROR {image_path.name}: {error}")

    pd.DataFrame(records).to_csv(
        OUTPUT_DIR / "segmentation_qc_report.csv",
        index=False,
    )

    pd.DataFrame(
        errors,
        columns=["image", "error"],
    ).to_csv(
        OUTPUT_DIR / "segmentation_errors.csv",
        index=False,
    )

    create_contact_sheet(records)

    print(f"\nProcessed: {len(records)}")
    print(f"Errors: {len(errors)}")
    print(f"Outputs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
