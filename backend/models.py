"""
API Models for Deep Fake Detection System
"""
from pydantic import BaseModel
from typing import Dict, Optional, Any
from enum import Enum

class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"

class DetectionSettings(BaseModel):
    use_ensemble: bool = True
    use_efficientnet: bool = True
    use_mobilenet: bool = True
    use_frequency: bool = True
    use_face_analysis: bool = True
    confidence_threshold: float = 0.5
    max_faces: int = 5
    frame_skip: int = 5
    max_frames: int = 50

class MediaMetadata(BaseModel):
    width: int
    height: int
    format: str
    size_bytes: int

class ModelPrediction(BaseModel):
    confidence: float
    is_fake: bool
    processing_time: float

class DetectionPredictions(BaseModel):
    real_ai_openai: Optional[ModelPrediction] = None
    ensemble: Optional[ModelPrediction] = None
    efficientnet: Optional[ModelPrediction] = None
    mobilenet: Optional[ModelPrediction] = None
    frequency: Optional[ModelPrediction] = None
    face: Optional[ModelPrediction] = None

class DetectionSummary(BaseModel):
    is_fake: bool
    confidence_score: float
    total_processing_time: float

class DetectionResponse(BaseModel):
    success: bool
    meta: MediaMetadata
    predictions: DetectionPredictions
    summary: DetectionSummary
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    models_available: Dict[str, bool]

class ModelsResponse(BaseModel):
    available_models: Dict[str, bool]
    default_settings: DetectionSettings