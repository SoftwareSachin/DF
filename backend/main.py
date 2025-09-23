"""
FastAPI Backend for Deep Fake Detection System
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
import json
import logging
from typing import Optional
import time
from pathlib import Path

from .models import (
    DetectionResponse, DetectionSettings, HealthResponse, 
    ModelsResponse, MediaMetadata, DetectionPredictions, 
    DetectionSummary, ModelPrediction
)
from .detector import DeepFakeDetectionService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Deep Fake Detection API", version="1.0.0")

# CORS configuration for Replit environment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Replit proxy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detection service
detection_service = DeepFakeDetectionService()

# Serve React frontend
frontend_path = Path(__file__).parent.parent / "frontend" / "dist"

# Mount static files first (with higher priority)
if frontend_path.exists():
    # Serve static assets
    app.mount("/assets", StaticFiles(directory=str(frontend_path / "assets")), name="assets")
    # Serve other static files
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

@app.get("/")
async def serve_frontend():
    """Serve React frontend"""
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Frontend not built. Run: npm run build in frontend directory"}

@app.get("/favicon.ico")
async def favicon():
    """Serve favicon"""
    favicon_path = frontend_path / "vite.svg"  # Use vite.svg as favicon
    if favicon_path.exists():
        return FileResponse(str(favicon_path))
    return {"message": "Favicon not found"}

@app.get("/vite.svg")
async def vite_logo():
    """Serve vite logo"""
    logo_path = frontend_path / "vite.svg"
    if logo_path.exists():
        return FileResponse(str(logo_path))
    return {"message": "Logo not found"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_status = detection_service.get_models_status()
    return HealthResponse(
        status="healthy" if any(models_status.values()) else "degraded",
        models_available=models_status
    )

@app.get("/api/models", response_model=ModelsResponse)
async def get_models():
    """Get available models and default settings"""
    return ModelsResponse(
        available_models=detection_service.get_models_status(),
        default_settings=DetectionSettings()
    )

@app.post("/api/analyze/image", response_model=DetectionResponse)
async def analyze_image(
    file: UploadFile = File(...),
    settings: str = Form(...)
):
    """Analyze image for deep fake detection"""
    try:
        # Parse settings
        detection_settings = DetectionSettings.parse_raw(settings)
        
        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file data
        file_data = await file.read()
        
        # Process with detection service
        result = await detection_service.analyze_image(
            file_data, file.filename or "unknown", detection_settings
        )
        
        return result
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid settings JSON")
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze/video", response_model=DetectionResponse)
async def analyze_video(
    file: UploadFile = File(...),
    settings: str = Form(...)
):
    """Analyze video for deep fake detection"""
    try:
        # Parse settings
        detection_settings = DetectionSettings.parse_raw(settings)
        
        # Validate file
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Read file data
        file_data = await file.read()
        
        # Process with detection service
        result = await detection_service.analyze_video(
            file_data, file.filename or "unknown", detection_settings
        )
        
        return result
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid settings JSON")
    except Exception as e:
        logger.error(f"Video analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)