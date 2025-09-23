"""
Deep Fake Detection Service
"""
import numpy as np
import cv2
import tempfile
import os
from typing import Dict, Optional, Any
import time
import logging
from PIL import Image
from io import BytesIO

from .simple_detector import SimpleDetector, MockOpenAIDetector

from .models import (
    DetectionResponse, DetectionSettings, MediaMetadata,
    DetectionPredictions, DetectionSummary, ModelPrediction
)

logger = logging.getLogger(__name__)

class DeepFakeDetectionService:
    """Service for deep fake detection using multiple AI models"""
    
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_IMAGE_DIMENSION = 2048
    
    def __init__(self):
        """Initialize detection models"""
        try:
            # Load Haar cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Initialize simplified detectors
            self.ensemble_detector = SimpleDetector("ensemble")
            self.efficientnet_detector = SimpleDetector("efficientnet")
            self.mobilenet_detector = SimpleDetector("mobilenet")
            self.frequency_analyzer = SimpleDetector("frequency")
            self.face_analyzer = SimpleDetector("face")
            self.openai_detector = MockOpenAIDetector()
            
            logger.info("Detection models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            
    def get_models_status(self) -> Dict[str, bool]:
        """Get status of available models"""
        return {
            "ensemble": hasattr(self, 'ensemble_detector'),
            "efficientnet": hasattr(self, 'efficientnet_detector'),
            "mobilenet": hasattr(self, 'mobilenet_detector'),
            "frequency": hasattr(self, 'frequency_analyzer'),
            "face": hasattr(self, 'face_analyzer'),
            "openai": hasattr(self, 'openai_detector')
        }
    
    def _validate_file_security(self, file_data: bytes) -> Dict[str, Any]:
        """Validate file for security and size constraints"""
        try:
            if len(file_data) > self.MAX_FILE_SIZE:
                return {'valid': False, 'error': f'File size exceeds maximum limit of {self.MAX_FILE_SIZE // (1024*1024)}MB'}
            
            try:
                pil_image = Image.open(BytesIO(file_data))
                if pil_image.width > self.MAX_IMAGE_DIMENSION or pil_image.height > self.MAX_IMAGE_DIMENSION:
                    return {'valid': False, 'error': f'Image dimensions exceed maximum limit of {self.MAX_IMAGE_DIMENSION}px'}
                
                return {
                    'valid': True, 
                    'format': pil_image.format, 
                    'size': (pil_image.width, pil_image.height),
                    'bytes': len(file_data)
                }
            except Exception:
                return {'valid': False, 'error': 'Invalid image format or corrupted file'}
                
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            return {'valid': False, 'error': 'Failed to validate file'}
    
    async def analyze_image(self, file_data: bytes, filename: str, settings: DetectionSettings) -> DetectionResponse:
        """Analyze image for deep fake detection"""
        start_time = time.time()
        
        try:
            # Validate file
            validation = self._validate_file_security(file_data)
            if not validation['valid']:
                return DetectionResponse(
                    success=False,
                    error=validation['error'],
                    meta=MediaMetadata(width=0, height=0, format="", size_bytes=0),
                    predictions=DetectionPredictions(),
                    summary=DetectionSummary(is_fake=False, confidence_score=0.0, total_processing_time=0.0)
                )
            
            # Create metadata
            meta = MediaMetadata(
                width=validation['size'][0],
                height=validation['size'][1],
                format=validation['format'],
                size_bytes=validation['bytes']
            )
            
            # Convert to OpenCV format
            nparr = np.frombuffer(file_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return DetectionResponse(
                    success=False,
                    error="Failed to decode image",
                    meta=meta,
                    predictions=DetectionPredictions(),
                    summary=DetectionSummary(is_fake=False, confidence_score=0.0, total_processing_time=0.0)
                )
            
            predictions = DetectionPredictions()
            prediction_scores = []
            
            # Run ensemble detection if enabled
            if settings.use_ensemble and hasattr(self, 'ensemble_detector'):
                try:
                    model_start = time.time()
                    if hasattr(self.ensemble_detector, 'predict'):
                        result = self.ensemble_detector.predict(image)
                        processing_time = time.time() - model_start
                        
                        predictions.ensemble = ModelPrediction(
                            confidence=float(result['confidence']),
                            is_fake=bool(result['confidence'] > settings.confidence_threshold),
                            processing_time=processing_time
                        )
                        prediction_scores.append(float(result['confidence']))
                except Exception as e:
                    logger.warning(f"Ensemble detection failed: {e}")
            
            # Run EfficientNet detection if enabled
            if settings.use_efficientnet and hasattr(self, 'efficientnet_detector'):
                try:
                    model_start = time.time()
                    if hasattr(self.efficientnet_detector, 'predict'):
                        result = self.efficientnet_detector.predict(image)
                        processing_time = time.time() - model_start
                        
                        predictions.efficientnet = ModelPrediction(
                            confidence=float(result['confidence']),
                            is_fake=bool(result['confidence'] > settings.confidence_threshold),
                            processing_time=processing_time
                        )
                        prediction_scores.append(float(result['confidence']))
                except Exception as e:
                    logger.warning(f"EfficientNet detection failed: {e}")
            
            # Run MobileNet detection if enabled
            if settings.use_mobilenet and hasattr(self, 'mobilenet_detector'):
                try:
                    model_start = time.time()
                    if hasattr(self.mobilenet_detector, 'predict'):
                        result = self.mobilenet_detector.predict(image)
                        processing_time = time.time() - model_start
                        
                        predictions.mobilenet = ModelPrediction(
                            confidence=float(result['confidence']),
                            is_fake=bool(result['confidence'] > settings.confidence_threshold),
                            processing_time=processing_time
                        )
                        prediction_scores.append(float(result['confidence']))
                except Exception as e:
                    logger.warning(f"MobileNet detection failed: {e}")
            
            # Run frequency analysis if enabled
            if settings.use_frequency and hasattr(self, 'frequency_analyzer'):
                try:
                    model_start = time.time()
                    if hasattr(self.frequency_analyzer, 'analyze'):
                        result = self.frequency_analyzer.analyze(image)
                        processing_time = time.time() - model_start
                        
                        predictions.frequency = ModelPrediction(
                            confidence=float(result['confidence']),
                            is_fake=bool(result['confidence'] > settings.confidence_threshold),
                            processing_time=processing_time
                        )
                        prediction_scores.append(float(result['confidence']))
                except Exception as e:
                    logger.warning(f"Frequency analysis failed: {e}")
            
            # Run face analysis if enabled
            if settings.use_face_analysis and hasattr(self, 'face_analyzer'):
                try:
                    model_start = time.time()
                    if hasattr(self.face_analyzer, 'analyze'):
                        result = self.face_analyzer.analyze(image, max_faces=settings.max_faces)
                        processing_time = time.time() - model_start
                        
                        predictions.face = ModelPrediction(
                            confidence=float(result['confidence']),
                            is_fake=bool(result['confidence'] > settings.confidence_threshold),
                            processing_time=processing_time
                        )
                        prediction_scores.append(float(result['confidence']))
                except Exception as e:
                    logger.warning(f"Face analysis failed: {e}")
            
            # Run OpenAI detection if available
            if hasattr(self, 'openai_detector'):
                try:
                    model_start = time.time()
                    if hasattr(self.openai_detector, 'analyze_image_bytes'):
                        result = self.openai_detector.analyze_image_bytes(file_data, filename)
                        processing_time = time.time() - model_start
                        
                        if 'confidence' in result:
                            predictions.real_ai_openai = ModelPrediction(
                                confidence=float(result['confidence']),
                                is_fake=bool(result['confidence'] > settings.confidence_threshold),
                                processing_time=processing_time
                            )
                            prediction_scores.append(float(result['confidence']))
                except Exception as e:
                    logger.warning(f"OpenAI detection failed: {e}")
            
            # Calculate summary
            avg_confidence = np.mean(prediction_scores) if prediction_scores else 0.0
            total_time = time.time() - start_time
            
            summary = DetectionSummary(
                is_fake=avg_confidence > settings.confidence_threshold,
                confidence_score=float(avg_confidence),
                total_processing_time=total_time
            )
            
            return DetectionResponse(
                success=True,
                meta=meta,
                predictions=predictions,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            total_time = time.time() - start_time
            return DetectionResponse(
                success=False,
                error=str(e),
                meta=MediaMetadata(width=0, height=0, format="", size_bytes=0),
                predictions=DetectionPredictions(),
                summary=DetectionSummary(is_fake=False, confidence_score=0.0, total_processing_time=total_time)
            )
    
    async def analyze_video(self, file_data: bytes, filename: str, settings: DetectionSettings) -> DetectionResponse:
        """Analyze video for deep fake detection"""
        start_time = time.time()
        
        try:
            # Save video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(file_data)
                video_path = tmp_file.name
            
            try:
                # Process video frames
                cap = cv2.VideoCapture(video_path)
                frame_predictions = []
                frame_count = 0
                
                while cap.isOpened() and frame_count < settings.max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % settings.frame_skip == 0:
                        # Convert frame to bytes for analysis
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_bytes = buffer.tobytes()
                        
                        # Analyze frame
                        frame_result = await self.analyze_image(frame_bytes, f"{filename}_frame_{frame_count}", settings)
                        if frame_result.success:
                            frame_predictions.append(frame_result.summary.confidence_score)
                    
                    frame_count += 1
                
                cap.release()
                
                # Calculate video-level summary
                if frame_predictions:
                    avg_confidence = np.mean(frame_predictions)
                    meta = MediaMetadata(
                        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.get(cv2.CAP_PROP_FRAME_WIDTH) else 0,
                        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.get(cv2.CAP_PROP_FRAME_HEIGHT) else 0,
                        format="video",
                        size_bytes=len(file_data)
                    )
                else:
                    avg_confidence = 0.0
                    meta = MediaMetadata(width=0, height=0, format="video", size_bytes=len(file_data))
                
                total_time = time.time() - start_time
                
                return DetectionResponse(
                    success=True,
                    meta=meta,
                    predictions=DetectionPredictions(),  # Individual frame predictions not returned for simplicity
                    summary=DetectionSummary(
                        is_fake=avg_confidence > settings.confidence_threshold,
                        confidence_score=float(avg_confidence),
                        total_processing_time=total_time
                    )
                )
                
            finally:
                # Clean up temporary file
                if os.path.exists(video_path):
                    os.unlink(video_path)
                    
        except Exception as e:
            logger.error(f"Video analysis error: {e}")
            total_time = time.time() - start_time
            return DetectionResponse(
                success=False,
                error=str(e),
                meta=MediaMetadata(width=0, height=0, format="video", size_bytes=len(file_data)),
                predictions=DetectionPredictions(),
                summary=DetectionSummary(is_fake=False, confidence_score=0.0, total_processing_time=total_time)
            )