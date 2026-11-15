"""
Animal Metrology CV - FastAPI RESTful Service
============================================
Exposes endpoints for uploading animal images, performing YOLO segmentation & pose
keypoint metrology, and returning real-time morphological measurements.
"""

from __future__ import annotations

import os
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Security,
    status,
    Query,
    Request,
    Response,
)
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.metrology import AnimalMetrologyPipeline

load_dotenv()

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MetrologyAPI")

# Security and limits
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
MAX_FILE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
API_KEY_NAME = "X-API-Key"
DEMO_API_KEY = os.getenv("API_KEY", "metrology-demo-key-2026")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Initialize FastAPI application with Swagger metadata
app = FastAPI(
    title="Animal Metrology CV API",
    description=(
        "RESTful API for animal segmentation, eye keypoint detection, and metric metrology. "
        "Supports single-animal interocular distance and cross-individual right-eye distance computation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")

# CORS Middleware (Restricted origins by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Swagger UI (/docs) and ReDoc require CDN assets from jsdelivr and inline script/style
    if request.url.path in ("/docs", "/redoc", "/openapi.json"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https://fastapi.tiangolo.com; "
            "frame-ancestors 'none';"
        )
    else:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; frame-ancestors 'none';"
        )
    return response


# Global pipeline instance (lazy or initial load)
pipeline = AnimalMetrologyPipeline()


# ==============================================================================
# Pydantic Schemas for Swagger Documentation
# ==============================================================================
class EyeCoordinate(BaseModel):
    x: Optional[float] = Field(None, description="X pixel coordinate")
    y: Optional[float] = Field(None, description="Y pixel coordinate")
    conf: Optional[float] = Field(None, description="Keypoint confidence score")
    inside_mask: Optional[bool] = Field(
        None, description="Whether the predicted keypoint lies inside its instance mask"
    )


class AnimalMeasurement(BaseModel):
    animal_id: str
    class_name: str
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2] bounding box")
    bbox_confidence: float
    mask_polygon: Optional[List[List[float]]] = None
    pose_candidate_left_eye: Optional[EyeCoordinate] = Field(
        None, description="Raw pose candidate; may lie outside the instance mask"
    )
    pose_candidate_right_eye: Optional[EyeCoordinate] = Field(
        None, description="Raw pose candidate; may lie outside the instance mask"
    )
    pose_candidate_interocular_dist_px: Optional[float] = Field(
        None, description="Unverified distance between raw pose candidates in pixels"
    )
    pose_candidate_interocular_dist_mm: Optional[float] = Field(
        None, description="Unverified distance between raw pose candidates in millimeters"
    )
    left_eye: Optional[EyeCoordinate] = Field(
        None, description="Pose estimate that passed confidence and instance-mask checks"
    )
    right_eye: Optional[EyeCoordinate] = Field(
        None, description="Pose estimate that passed confidence and instance-mask checks"
    )
    mask_area_px: float = Field(..., description="Segmented animal mask area in pixels")
    interocular_dist_px: Optional[float] = Field(
        None, description="Task 3: Distance from mask-checked pose estimates in pixels"
    )
    interocular_dist_mm: Optional[float] = Field(
        None, description="Task 3: Distance from mask-checked pose estimates in millimeters"
    )


class PairwiseMeasurement(BaseModel):
    image_id: str
    animal_1_id: str
    animal_2_id: str
    animal_1_left_eye: str
    animal_1_right_eye: str
    animal_1_interocular_dist_mm: Optional[float]
    animal_2_interocular_dist_mm: Optional[float]
    right_eye_to_right_eye_dist_px: Optional[float] = Field(
        None, description="Task 4: Distance between mask-checked pose estimates in pixels"
    )
    right_eye_to_right_eye_dist_mm: Optional[float] = Field(
        None, description="Task 4: Distance between mask-checked pose estimates in mm"
    )
    pose_candidate_animal_1_right_eye: Optional[EyeCoordinate] = Field(
        None, description="Unverified raw right-eye candidate for animal 1"
    )
    pose_candidate_animal_2_right_eye: Optional[EyeCoordinate] = Field(
        None, description="Unverified raw right-eye candidate for animal 2"
    )
    pose_candidate_right_eye_to_right_eye_dist_px: Optional[float] = Field(
        None, description="Unverified raw right-eye distance in pixels"
    )
    pose_candidate_right_eye_to_right_eye_dist_mm: Optional[float] = Field(
        None, description="Unverified raw right-eye distance in millimeters"
    )


class PredictResponse(BaseModel):
    success: bool
    image_id: str
    image_shape: Dict[str, int]
    pixel_to_mm_scale: float
    detected_animals_count: int
    animals: List[AnimalMeasurement]
    pairwise_measurements: List[PairwiseMeasurement]
    csv_saved_path: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_weights: str
    pose_model_loaded: bool
    pose_model_weights: str
    pixel_to_mm_scale: float


# ==============================================================================
# Authentication Helper
# ==============================================================================
async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> bool:
    """
    Validate incoming API Key header.
    Allows demo key 'metrology-demo-key-2026' or configured env variable.
    """
    if not api_key:
        # Allow open access if explicitly configured or inform caller
        return True
    if api_key != DEMO_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API Key.",
        )
    return True


# ==============================================================================
# API Endpoints
# ==============================================================================
@app.get("/", summary="API Root", tags=["System"])
async def root():
    return {
        "service": "Animal Metrology CV Service",
        "version": "1.0.0",
        "ui": "/ui/",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check", tags=["System"])
async def health():
    return {
        "status": "healthy" if pipeline.seg_model is not None and pipeline.pose_model is not None else "unavailable",
        "model_loaded": pipeline.seg_model is not None,
        "model_weights": pipeline.seg_model_weights,
        "pose_model_loaded": pipeline.pose_model is not None,
        "pose_model_weights": pipeline.pose_model_weights,
        "pixel_to_mm_scale": pipeline.pixel_to_mm_scale,
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Detect Animals and Return Measurements (JSON)",
    tags=["Metrology"],
)
async def predict_metrology(
    file: UploadFile = File(..., description="Animal image file (JPEG, PNG, or WEBP)"),
    pixel_to_mm_scale: Optional[float] = Query(
        None,
        description="Override metric conversion scale (mm/px). Defaults to server .env setting (0.15).",
        gt=0.0,
    ),
    save_to_csv: bool = Query(
        True,
        description="Whether to append the measurements to outputs/measurement_results.csv",
    ),
    authenticated: bool = Security(verify_api_key),
):
    """
    Upload an image containing one or multiple animals to perform:
    1. Instance segmentation and dual-eye keypoint localization.
    2. Task 3: Single-animal interocular distance measurement (pixel and mm).
    3. Task 4: Cross-individual distance between right eyes of arbitrary animal pairs.

    Returns detection details and measurements as JSON. Use /predict/visualize
    when you want the annotated image instead.
    """
    # 1. MIME type validation
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file.content_type}'. Allowed: {list(ALLOWED_MIME_TYPES)}",
        )

    # 2. File size verification to prevent DoS
    contents = await file.read()
    if len(contents) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB}MB.",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # 3. Sanitize and isolate image identification
    original_filename = Path(file.filename or "upload.jpg").name
    safe_image_id = f"{uuid.uuid4().hex[:8]}_{original_filename}"

    # 4. Execute metrology pipeline
    try:
        # Optionally override scale for this request
        if pixel_to_mm_scale is not None:
            pipeline.pixel_to_mm_scale = pixel_to_mm_scale

        results = pipeline.infer(contents, image_id=safe_image_id)

        # 5. Optionally export/append to CSV
        csv_path_str = None
        if save_to_csv and results["pairwise_measurements"]:
            csv_path = pipeline.export_to_csv(
                results["pairwise_measurements"],
                append=True,
            )
            csv_path_str = str(csv_path)

        return {
            "success": True,
            "image_id": safe_image_id,
            "image_shape": results["image_shape"],
            "pixel_to_mm_scale": pipeline.pixel_to_mm_scale,
            "detected_animals_count": results["detected_animals_count"],
            "animals": results["animals"],
            "pairwise_measurements": results["pairwise_measurements"],
            "csv_saved_path": csv_path_str,
        }

    except ValueError as val_err:
        logger.error("Image decode error: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image data: {str(val_err)}",
        )
    except Exception as exc:
        logger.exception("Metrology inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the metrology request.",
        )


@app.post(
    "/predict/visualize",
    summary="Return Annotated Detection Image (JPEG)",
    tags=["Metrology"],
    response_class=Response,
    description=(
        "Returns only the annotated JPEG image. To view animal detections, eye "
        "coordinates, and measurements as JSON, use POST /predict."
    ),
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "Returns annotated JPEG image with segmented masks, eye keypoints, and measurement distance lines.",
        }
    },
)
async def predict_and_visualize(
    file: UploadFile = File(..., description="Animal image file (JPEG, PNG, or WEBP)"),
    pixel_to_mm_scale: Optional[float] = Query(
        None,
        description="Override metric conversion scale (mm/px). Defaults to server .env setting (0.15).",
        gt=0.0,
    ),
    authenticated: bool = Security(verify_api_key),
):
    """
    Upload an animal image to directly receive the visualized image:
    - Translucent green contour mask for each animal (YOLOv8-seg)
    - Blue/Red keypoint markers for Left/Right eyes
    - Yellow line indicating single-animal eye distance (Task 3)
    - Magenta line indicating cross-individual right-eye distance (Task 4)
    """
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file.content_type}'. Allowed: {list(ALLOWED_MIME_TYPES)}",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB}MB.",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    original_filename = Path(file.filename or "upload.jpg").name
    safe_image_id = f"vis_{uuid.uuid4().hex[:8]}_{original_filename}"

    try:
        if pixel_to_mm_scale is not None:
            pipeline.pixel_to_mm_scale = pixel_to_mm_scale

        results = pipeline.infer(contents, image_id=safe_image_id)
        annotated_bgr = pipeline.annotate_image(contents, results)

        # Encode to JPEG in-memory
        success, buffer = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            raise ValueError("Failed to encode annotated image to JPEG buffer.")

        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except ValueError as val_err:
        logger.error("Visualization error: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image: {str(val_err)}",
        )
    except Exception as exc:
        logger.exception("Visualization pipeline error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the visualization image.",
        )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    logger.info("Starting Animal Metrology API on http://%s:%d", host, port)
    uvicorn.run("src.api:app", host=host, port=port, reload=True)
