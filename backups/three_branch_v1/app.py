
from pathlib import Path
import tempfile
import traceback

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from predict import predict
from segment_paneer import make_candidates, score_contour
from flash_roi_pipeline import align_flash_to_ambient


# ============================================================
# CONFIGURATION
# ============================================================

register_heif_opener()

ROOT = Path(__file__).resolve().parent
DEBUG_DIR = ROOT / "results"

app = FastAPI(
    title="AnalogueNet API",
    description="Experimental paneer image and video classification",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif"
}

ALLOWED_VIDEO_EXTENSIONS = {
    ".mov", ".mp4", ".avi", ".m4v"
}

MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_VIDEO_BYTES = 150 * 1024 * 1024


# ============================================================
# FILE HELPERS
# ============================================================

async def save_upload(
    upload: UploadFile,
    destination: Path,
    max_bytes: int,
):
    size = 0

    try:
        with destination.open("wb") as output:
            while True:
                chunk = await upload.read(1024 * 1024)

                if not chunk:
                    break

                size += len(chunk)

                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"{upload.filename} exceeds the "
                            "upload size limit."
                        ),
                    )

                output.write(chunk)
    finally:
        await upload.close()

    if size == 0:
        raise HTTPException(
            status_code=400,
            detail=f"{upload.filename} is empty.",
        )


def convert_heic_if_needed(path: Path) -> Path:
    """Convert HEIC/HEIF images into OpenCV-readable JPEG files."""

    if path.suffix.lower() not in {".heic", ".heif"}:
        return path

    converted_path = path.with_suffix(".jpg")

    try:
        with Image.open(path) as image:
            image.convert("RGB").save(
                converted_path,
                "JPEG",
                quality=95,
            )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=422,
            detail=f"HEIC conversion failed: {exc}",
        ) from exc

    return converted_path


def validate_extension(
    upload: UploadFile,
    allowed: set[str],
):
    extension = Path(upload.filename or "").suffix.lower()

    if extension not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {upload.filename}",
        )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def create_difference_image(
    ambient: np.ndarray,
    flash: np.ndarray,
) -> np.ndarray:
    """
    Align flash to ambient, isolate the strongest paneer candidate,
    and return a grayscale difference crop.

    This is an initial inference approximation. It has not been
    verified as identical to the original training pipeline.
    """

    gray = cv2.cvtColor(
        ambient,
        cv2.COLOR_BGR2GRAY,
    )

    image_area = gray.shape[0] * gray.shape[1]
    candidates = []

    # Find candidate paneer contours.
    for method, threshold in make_candidates(gray):
        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            metrics = score_contour(
                contour,
                gray,
                image_area,
            )

            if metrics is not None:
                candidates.append({
                    **metrics,
                    "method": method,
                    "contour": contour,
                })

    if not candidates:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not isolate the paneer. Place the whole piece "
                "on a contrasting background and take a clearer photo."
            ),
        )

    # Select the highest-scoring contour.
    best = max(
        candidates,
        key=lambda item: item["score"],
    )

    x, y, w, h = (
        best["x"],
        best["y"],
        best["w"],
        best["h"],
    )

    # Create and slightly erode the paneer mask.
    mask = np.zeros(
        gray.shape,
        dtype=np.uint8,
    )

    cv2.drawContours(
        mask,
        [best["contour"]],
        -1,
        255,
        cv2.FILLED,
    )

    mask = cv2.erode(
        mask,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    )

    # Align flash to ambient.
    try:
        aligned_flash, valid_mask, _, _, _ = (
            align_flash_to_ambient(
                ambient,
                flash,
            )
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=422,
            detail=f"Image alignment failed: {exc}",
        ) from exc

    # Convert aligned images to grayscale.
    ambient_gray = gray

    flash_gray = cv2.cvtColor(
        aligned_flash,
        cv2.COLOR_BGR2GRAY,
    )

    # Calculate absolute pixel difference.
    difference = cv2.absdiff(
        ambient_gray,
        flash_gray,
    )

    # Retain only paneer pixels overlapping valid flash pixels.
    valid_paneer = (
        (mask > 0)
        & (valid_mask > 0)
    )

    difference[~valid_paneer] = 0

    # Crop paneer region with 10-pixel padding.
    padding = 10
    height, width = gray.shape

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)

    difference_crop = difference[y1:y2, x1:x2]

    if difference_crop.size == 0:
        raise HTTPException(
            status_code=422,
            detail="The paneer difference crop is empty.",
        )

    return difference_crop


# ============================================================
# MODEL INFERENCE
# ============================================================

def run_inference(
    difference_path: Path,
    video_path: Path,
):
    try:
        result = predict(
            difference_image=difference_path,
            press_video=video_path,
        )

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print("Model inference failed:")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Check the backend terminal.",
        ) from exc

    return {
        "status": "success",
        "prediction": result["predicted_class"],
        "scores": {
            "untreated": result["untreated_score"],
            "starch_treated": result["starch_treated_score"],
        },
        "experimental": True,
        "warning": (
            "Research prototype only. This result is not a validated "
            "food-safety or quality determination."
        ),
    }


# ============================================================
# HEALTH ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "name": "AnalogueNet API",
        "status": "running",
        "purpose": "Experimental paneer classification",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "multimodal_fusion_v1",
    }


# ============================================================
# EXISTING ENDPOINT: PREPROCESSED IMAGE + VIDEO
# ============================================================

@app.post("/predict")
async def predict_paneer(
    difference_image: UploadFile = File(...),
    video: UploadFile = File(...),
):
    validate_extension(
        difference_image,
        ALLOWED_IMAGE_EXTENSIONS,
    )

    validate_extension(
        video,
        ALLOWED_VIDEO_EXTENSIONS,
    )

    with tempfile.TemporaryDirectory(
        prefix="analoguenet_predict_"
    ) as temp_dir:

        folder = Path(temp_dir)

        image_ext = Path(
            difference_image.filename or ""
        ).suffix.lower()

        video_ext = Path(
            video.filename or ""
        ).suffix.lower()

        image_path = folder / f"difference{image_ext}"
        video_path = folder / f"press{video_ext}"

        await save_upload(
            difference_image,
            image_path,
            MAX_IMAGE_BYTES,
        )

        await save_upload(
            video,
            video_path,
            MAX_VIDEO_BYTES,
        )

        image_path = convert_heic_if_needed(image_path)

        return run_inference(
            image_path,
            video_path,
        )


# ============================================================
# WEBSITE ENDPOINT: AMBIENT + FLASH + PRESSING VIDEO
# ============================================================

@app.post("/scan")
async def scan_paneer(
    ambient_image: UploadFile = File(...),
    flash_image: UploadFile = File(...),
    video: UploadFile = File(...),
):
    validate_extension(
        ambient_image,
        ALLOWED_IMAGE_EXTENSIONS,
    )

    validate_extension(
        flash_image,
        ALLOWED_IMAGE_EXTENSIONS,
    )

    validate_extension(
        video,
        ALLOWED_VIDEO_EXTENSIONS,
    )

    with tempfile.TemporaryDirectory(
        prefix="analoguenet_scan_"
    ) as temp_dir:

        folder = Path(temp_dir)

        uploads = [
            (ambient_image, "ambient"),
            (flash_image, "flash"),
            (video, "press"),
        ]

        paths = {}

        for upload, name in uploads:
            extension = Path(
                upload.filename or ""
            ).suffix.lower()

            path = folder / f"{name}{extension}"

            limit = (
                MAX_VIDEO_BYTES
                if name == "press"
                else MAX_IMAGE_BYTES
            )

            await save_upload(
                upload,
                path,
                limit,
            )

            paths[name] = path

        # Convert iPhone HEIC/HEIF images when necessary.
        ambient_path = convert_heic_if_needed(paths["ambient"])
        flash_path = convert_heic_if_needed(paths["flash"])
        video_path = paths["press"]

        ambient = cv2.imread(str(ambient_path))
        flash = cv2.imread(str(flash_path))

        if ambient is None or flash is None:
            raise HTTPException(
                status_code=422,
                detail="Could not decode one or both uploaded photos.",
            )

        if ambient.size == 0 or flash.size == 0:
            raise HTTPException(
                status_code=422,
                detail="An uploaded photo is empty.",
            )

        # Generate the difference image.
        difference = create_difference_image(
            ambient,
            flash,
        )

        # Save a permanent debugging copy.
        DEBUG_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        debug_path = DEBUG_DIR / "website_debug_difference.jpg"

        if not cv2.imwrite(str(debug_path), difference):
            raise HTTPException(
                status_code=500,
                detail="Could not save the debugging difference image.",
            )

        # Save a temporary copy for inference.
        difference_path = folder / "difference.jpg"

        if not cv2.imwrite(str(difference_path), difference):
            raise HTTPException(
                status_code=500,
                detail="Could not create the difference image.",
            )

        # Run the multimodal model.
        result = run_inference(
            difference_path,
            video_path,
        )

        result["preprocessing_note"] = (
            "Raw-photo preprocessing is an initial approximation and "
            "has not been verified against the original training-image "
            "generation pipeline."
        )

        return result
