"""
Genuine Deep Fake Detection System - Production-Ready AI Application
Advanced image and video analysis platform with state-of-the-art deep fake detection models.
"""
import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from typing import Dict, List, Tuple, Optional
import time
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from PIL import Image
import base64
from io import BytesIO
import logging
import sys
from deepfake_models import (
    EnsembleDeepFakeDetector,
    EfficientNetDeepFakeDetector,
    MobileNetDeepFakeDetector,
    FrequencyDomainAnalyzer,
    MediaPipeFaceAnalyzer
)

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Genuine Deep Fake Detection System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        background: #ffffff;
        border: 2px solid #e0e6ed;
        padding: 1rem;
        margin-bottom: 1rem;
        text-align: center;
        border-radius: 4px;
    }
    .main-header h1 {
        color: #2c3e50;
        font-weight: 600;
        margin: 0;
        font-size: 2.2rem;
    }
    .main-header p {
        color: #334155;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        font-weight: 400;
    }
    .detection-result {
        padding: 1.5rem;
        margin: 1rem 0;
        background: #ffffff;
        border: 1px solid #d1d9e0;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .analysis-section {
        background: #f8fafc;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3182ce;
        border-radius: 4px;
    }
    .fake-alert {
        background: #fed7d7;
        color: #742a2a;
        border: 1px solid #feb2b2;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        border-radius: 4px;
        border-left: 4px solid #e53e3e;
    }
    .real-alert {
        background: #c6f6d5;
        color: #22543d;
        border: 1px solid #9ae6b4;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        border-radius: 4px;
        border-left: 4px solid #38a169;
    }
    .model-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        background: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        font-size: 0.85rem;
        margin: 0.2rem;
        color: #4a5568;
        font-weight: 500;
    }
    .sidebar-section {
        background: #f8fafc;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 1rem;
        text-align: center;
    }
    .content-visibility {
        line-height: 1.6;
        font-size: 1rem;
        color: #334155;
    }
    .section-header {
        color: #2d3748;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

class GenuineDeepFakeDetectionSystem:
    """Production-ready genuine deep fake detection system with multiple AI models."""
    
    # Security constants
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB max file size
    MAX_IMAGE_DIMENSION = 2048  # Max image width/height
    ALLOWED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    ALLOWED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    def __init__(self):
        logger.info("Initializing Genuine Deep Fake Detection System")
        self.ensemble_detector = EnsembleDeepFakeDetector()
        self.supported_image_formats = self.ALLOWED_IMAGE_FORMATS
        self.supported_video_formats = self.ALLOWED_VIDEO_FORMATS
        
        # Initialize individual detectors for detailed analysis
        self.efficientnet_detector = EfficientNetDeepFakeDetector()
        self.mobilenet_detector = MobileNetDeepFakeDetector()
        self.frequency_analyzer = FrequencyDomainAnalyzer()
        self.face_analyzer = MediaPipeFaceAnalyzer()
    
    def render_header(self):
        """Render the application header."""
        st.markdown("""
        <div class="main-header">
            <h1>Deep Fake Detection System</h1>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with detection model options."""
        st.sidebar.title("Settings")
        
        # Media type selection
        media_type = st.sidebar.radio("Media Type", ["Image", "Video"], index=0)
        
        # Main options
        use_ensemble = st.sidebar.checkbox("Ensemble Detection (Recommended)", value=True)
        confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
        
        # Video specific options
        if media_type == "Video":
            frame_skip = st.sidebar.slider("Frame Skip (process every Nth frame)", 1, 30, 5)
            max_frames = st.sidebar.slider("Max Frames to Analyze", 10, 300, 50)
        else:
            frame_skip = 1
            max_frames = 1
        
        # Advanced options in expander
        with st.sidebar.expander("Advanced Options"):
            use_efficientnet = st.sidebar.checkbox("EfficientNet-B0", value=True)
            use_mobilenet = st.sidebar.checkbox("MobileNet-V2", value=True)
            use_frequency = st.sidebar.checkbox("Frequency Analysis", value=True)
            use_face_analysis = st.sidebar.checkbox("Face Landmark Analysis", value=True)
            max_faces = st.sidebar.slider("Max Faces to Analyze", 1, 10, 5)
        
        return {
            'media_type': media_type,
            'use_ensemble': use_ensemble,
            'use_efficientnet': use_efficientnet,
            'use_mobilenet': use_mobilenet,
            'use_frequency': use_frequency,
            'use_face_analysis': use_face_analysis,
            'confidence_threshold': confidence_threshold,
            'max_faces': max_faces,
            'frame_skip': frame_skip,
            'max_frames': max_frames
        }
    
    def validate_image_security(self, image_bytes: bytes) -> Dict[str, any]:
        """Validate image for security and size constraints."""
        try:
            # Check file size
            if len(image_bytes) > self.MAX_FILE_SIZE:
                return {'valid': False, 'error': f'File size exceeds maximum limit of {self.MAX_FILE_SIZE // (1024*1024)}MB'}
            
            # Check if it's a valid image
            try:
                pil_image = Image.open(BytesIO(image_bytes))
                # Check image dimensions
                if pil_image.width > self.MAX_IMAGE_DIMENSION or pil_image.height > self.MAX_IMAGE_DIMENSION:
                    return {'valid': False, 'error': f'Image dimensions exceed maximum limit of {self.MAX_IMAGE_DIMENSION}px'}
                
                return {'valid': True, 'format': pil_image.format, 'size': (pil_image.width, pil_image.height)}
            except Exception:
                return {'valid': False, 'error': 'Invalid image format or corrupted file'}
                
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            return {'valid': False, 'error': 'Failed to validate file'}
    
    def process_image_with_ai_bytes(self, image_bytes: bytes, filename: str, settings: Dict) -> Dict:
        """Process uploaded image with genuine AI deep fake detection."""
        try:
            logger.info(f"Processing image with AI models: {filename}")
            
            # Use provided image bytes
            
            # Security validation
            validation = self.validate_image_security(image_bytes)
            if not validation['valid']:
                logger.warning(f"Image validation failed: {validation['error']}")
                return {'error': validation['error']}
            
            # Convert to OpenCV format
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {'error': 'Could not decode image file'}
            
            results = {
                'image_shape': image.shape,
                'ai_predictions': {},
                'analysis_details': {}
            }
            
            # Ensemble detection (recommended)
            if settings['use_ensemble']:
                ensemble_result = self.ensemble_detector.detect_deepfake(image)
                results['ai_predictions']['ensemble'] = ensemble_result
            
            # Individual model predictions
            if settings['use_efficientnet']:
                efficientnet_result = self.efficientnet_detector.predict(image)
                results['ai_predictions']['efficientnet'] = efficientnet_result
            
            if settings['use_mobilenet']:
                mobilenet_result = self.mobilenet_detector.predict(image)
                results['ai_predictions']['mobilenet'] = mobilenet_result
            
            if settings['use_frequency']:
                frequency_result = self.frequency_analyzer.analyze_frequency_artifacts(image)
                results['ai_predictions']['frequency'] = frequency_result
                results['analysis_details']['frequency_analysis'] = frequency_result
            
            if settings['use_face_analysis']:
                face_result = self.face_analyzer.analyze_facial_landmarks(image)
                results['ai_predictions']['face_analysis'] = face_result
                results['analysis_details']['face_analysis'] = face_result
            
            # Calculate overall confidence
            results['overall_analysis'] = self._calculate_overall_confidence(results['ai_predictions'], settings)
            
            return results
            
        except Exception as e:
            logger.error(f"AI image processing error: {e}")
            return {'error': f'Error processing image with AI: {str(e)}'}
    
    def process_video_with_ai(self, video_bytes: bytes, filename: str, settings: Dict) -> Dict:
        """Process uploaded video with genuine AI deep fake detection."""
        try:
            logger.info(f"Processing video with AI models: {filename}")
            
            # Save video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                temp_video.write(video_bytes)
                temp_video_path = temp_video.name
            
            try:
                # Open video file
                cap = cv2.VideoCapture(temp_video_path)
                if not cap.isOpened():
                    return {'error': 'Could not open video file'}
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                frame_results = []
                frame_confidences = []
                frames_processed = 0
                frame_count = 0
                
                while cap.isOpened() and frames_processed < settings['max_frames']:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every Nth frame based on frame_skip setting
                    if frame_count % settings['frame_skip'] == 0:
                        # Process frame with AI models
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_bytes = buffer.tobytes()
                        
                        frame_result = self.process_image_with_ai_bytes(frame_bytes, f"frame_{frame_count}", settings)
                        
                        if 'error' not in frame_result:
                            frame_results.append({
                                'frame_number': frame_count,
                                'timestamp': frame_count / fps if fps > 0 else 0,
                                'analysis': frame_result
                            })
                            
                            overall_confidence = frame_result.get('overall_analysis', {}).get('overall_confidence', 0.5)
                            frame_confidences.append(overall_confidence)
                            
                        frames_processed += 1
                    
                    frame_count += 1
                
                cap.release()
                
                # Calculate video-level statistics
                if frame_confidences:
                    avg_confidence = np.mean(frame_confidences)
                    max_confidence = np.max(frame_confidences)
                    min_confidence = np.min(frame_confidences)
                    confidence_std = np.std(frame_confidences)
                    
                    # Determine if video contains deepfakes
                    suspicious_frames = sum(1 for c in frame_confidences if c > settings['confidence_threshold'])
                    suspicious_ratio = suspicious_frames / len(frame_confidences)
                    
                    video_is_fake = suspicious_ratio > 0.3  # If >30% of frames are suspicious
                    
                    results = {
                        'video_info': {
                            'total_frames': total_frames,
                            'fps': fps,
                            'frames_processed': frames_processed,
                            'frame_skip': settings['frame_skip']
                        },
                        'frame_results': frame_results,
                        'video_analysis': {
                            'avg_confidence': avg_confidence,
                            'max_confidence': max_confidence,
                            'min_confidence': min_confidence,
                            'confidence_std': confidence_std,
                            'suspicious_frames': suspicious_frames,
                            'suspicious_ratio': suspicious_ratio,
                            'is_deepfake': video_is_fake,
                            'overall_confidence': avg_confidence
                        }
                    }
                    
                    return results
                else:
                    return {'error': 'No frames could be processed successfully'}
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                    
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            return {'error': f'Error processing video with AI: {str(e)}'}
    
    def _calculate_overall_confidence(self, predictions: Dict, settings: Dict) -> Dict:
        """Calculate overall deep fake confidence from all models."""
        confidences = []
        model_results = []
        
        # Ensemble prediction (highest priority)
        if 'ensemble' in predictions and 'error' not in predictions['ensemble']:
            ensemble_conf = predictions['ensemble']['ensemble_prediction']['confidence']
            confidences.append(ensemble_conf)
            model_results.append({
                'model': 'Ensemble AI',
                'confidence': ensemble_conf,
                'weight': 0.5
            })
        
        # Individual model predictions
        for model_name, pred in predictions.items():
            if model_name == 'ensemble' or 'error' in pred:
                continue
                
            if model_name in ['efficientnet', 'mobilenet'] and 'confidence' in pred:
                confidences.append(pred['confidence'])
                model_results.append({
                    'model': pred.get('model', model_name),
                    'confidence': pred['confidence'],
                    'weight': 0.2
                })
            elif model_name == 'frequency' and 'artifacts_detected' in pred:
                freq_conf = pred['artifacts_detected'].get('confidence', 0.5)
                confidences.append(freq_conf)
                model_results.append({
                    'model': 'Frequency Analysis',
                    'confidence': freq_conf,
                    'weight': 0.15
                })
            elif model_name == 'face_analysis' and 'quality_metrics' in pred:
                face_conf = 1.0 - pred['quality_metrics'].get('avg_detection_confidence', 0.5)
                confidences.append(face_conf)
                model_results.append({
                    'model': 'Face Analysis',
                    'confidence': face_conf,
                    'weight': 0.15
                })
        
        # Calculate weighted average
        if confidences:
            overall_confidence = np.mean(confidences)
            is_deepfake = overall_confidence > settings['confidence_threshold']
        else:
            overall_confidence = 0.5
            is_deepfake = False
        
        return {
            'overall_confidence': float(overall_confidence),
            'is_deepfake': is_deepfake,
            'model_results': model_results,
            'total_models_used': len(model_results),
            'confidence_threshold': settings['confidence_threshold']
        }
    
    def render_ai_results(self, results: Dict, file_type: str):
        """Render AI-powered analysis results."""
        if 'error' in results:
            st.error(f"Error: {results['error']}")
            return
        
        overall_analysis = results.get('overall_analysis', {})
        confidence = overall_analysis.get('overall_confidence', 0.0)
        is_deepfake = overall_analysis.get('is_deepfake', False)
        
        # Main detection result
        if is_deepfake:
            st.error(f"DEEPFAKE DETECTED - Confidence: {confidence:.1%}")
        else:
            st.success(f"AUTHENTIC CONTENT - Confidence: {(1-confidence):.1%}")
        
        # Create tabs for detailed analysis
        tabs = st.tabs(["Summary", "Details"])
        
        # Summary tab
        with tabs[0]:
            self.render_summary_tab(results)
        
        # Details tab
        with tabs[1]:
            self.render_details_tab(results)
    
    def render_ai_predictions_tab(self, results: Dict):
        """Render AI predictions from all models."""
        st.markdown('<div class="section-header">AI Model Predictions</div>', unsafe_allow_html=True)
        
        predictions = results.get('ai_predictions', {})
        
        # Ensemble prediction
        if 'ensemble' in predictions:
            ensemble = predictions['ensemble']
            if 'error' not in ensemble:
                ensemble_pred = ensemble.get('ensemble_prediction', {})
                st.markdown('<div class="section-header">Ensemble AI Detection</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Ensemble Confidence", f"{ensemble_pred.get('confidence', 0):.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Prediction", "FAKE" if ensemble_pred.get('is_fake', False) else "REAL")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Inference Time", f"{ensemble_pred.get('total_inference_time', 0):.3f}s")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Individual model predictions
        st.markdown('<div class="section-header">Individual Model Results</div>', unsafe_allow_html=True)
        
        for model_name, pred in predictions.items():
            if model_name == 'ensemble' or 'error' in pred:
                continue
            
            with st.expander(f"{model_name.title()} Results"):
                if model_name in ['efficientnet', 'mobilenet']:
                    st.write(f"**Model**: {pred.get('model', model_name)}")
                    st.write(f"**Confidence**: {pred.get('confidence', 0):.3f}")
                    st.write(f"**Prediction**: {'FAKE' if pred.get('is_fake', False) else 'REAL'}")
                    st.write(f"**Inference Time**: {pred.get('inference_time', 0):.3f}s")
                elif model_name == 'frequency':
                    artifacts = pred.get('artifacts_detected', {})
                    st.write(f"**Anomaly Score**: {artifacts.get('anomaly_score', 0):.3f}")
                    st.write(f"**Artifacts Detected**: {'Yes' if artifacts.get('is_anomalous', False) else 'No'}")
                    st.write(f"**Confidence**: {artifacts.get('confidence', 0):.3f}")
                elif model_name == 'face_analysis':
                    st.write(f"**Faces Detected**: {pred.get('faces_detected', 0)}")
                    quality = pred.get('quality_metrics', {})
                    st.write(f"**Detection Confidence**: {quality.get('avg_detection_confidence', 0):.3f}")
    
    def render_summary_tab(self, results: Dict):
        """Render summary of key results."""
        overall_analysis = results.get('overall_analysis', {})
        model_results = overall_analysis.get('model_results', [])
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Confidence", f"{overall_analysis.get('overall_confidence', 0):.1%}")
        with col2:
            st.metric("Models Used", overall_analysis.get('total_models_used', 0))
        with col3:
            st.metric("Threshold", f"{overall_analysis.get('confidence_threshold', 0.5):.1%}")
        
        # Model comparison chart
        if model_results:
            models = [r['model'] for r in model_results]
            confidences = [r['confidence'] for r in model_results]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=models,
                y=confidences,
                marker_color='#3182ce',
                text=[f"{c:.1%}" for c in confidences],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Model Confidence Scores",
                xaxis_title="Model",
                yaxis_title="Confidence",
                yaxis=dict(range=[0, 1]),
                height=400
            )
            
            st.plotly_chart(fig, width='stretch')
    
    def render_technical_analysis_tab(self, results: Dict):
        """Render technical analysis details."""
        st.markdown('<div class="section-header">Technical Analysis</div>', unsafe_allow_html=True)
        
        analysis_details = results.get('analysis_details', {})
        
        # Frequency analysis
        if 'frequency_analysis' in analysis_details:
            freq_analysis = analysis_details['frequency_analysis']
            st.markdown('<div class="section-header">Frequency Domain Analysis</div>', unsafe_allow_html=True)
            
            features = freq_analysis.get('frequency_features', {})
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Frequency Features:**")
                for key, value in features.items():
                    if isinstance(value, (int, float)):
                        st.write(f"- {key.replace('_', ' ').title()}: {value:.4f}")
            
            with col2:
                artifacts = freq_analysis.get('artifacts_detected', {})
                st.write("**Artifact Detection:**")
                st.write(f"- Anomaly Score: {artifacts.get('anomaly_score', 0):.3f}")
                st.write(f"- Is Anomalous: {artifacts.get('is_anomalous', False)}")
        
        # Face analysis
        if 'face_analysis' in analysis_details:
            face_analysis = analysis_details['face_analysis']
            st.markdown('<div class="section-header">Facial Landmark Analysis</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Faces Detected**: {face_analysis.get('faces_detected', 0)}")
                quality = face_analysis.get('quality_metrics', {})
                st.write(f"**Average Detection Confidence**: {quality.get('avg_detection_confidence', 0):.3f}")
                st.write(f"**Total Landmarks**: {quality.get('total_landmarks_all_faces', 0)}")
            
            with col2:
                geometry = face_analysis.get('geometric_analysis', {})
                if geometry:
                    st.write("**Geometric Analysis:**")
                    for key, value in geometry.items():
                        if isinstance(value, (int, float)):
                            st.write(f"- {key.replace('_', ' ').title()}: {value:.4f}")
    
    def render_confidence_breakdown_tab(self, results: Dict):
        """Render confidence score breakdown."""
        st.markdown('<div class="section-header">Confidence Score Analysis</div>', unsafe_allow_html=True)
        
        overall_analysis = results.get('overall_analysis', {})
        
        # Overall metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Overall Confidence",
                f"{overall_analysis.get('overall_confidence', 0):.1%}",
                help="Combined confidence from all AI models"
            )
        
        with col2:
            st.metric(
                "Models Used",
                overall_analysis.get('total_models_used', 0),
                help="Number of AI models that analyzed this content"
            )
        
        with col3:
            st.metric(
                "Threshold",
                f"{overall_analysis.get('confidence_threshold', 0.5):.1%}",
                help="Confidence threshold for deepfake classification"
            )
        
        # Confidence distribution
        model_results = overall_analysis.get('model_results', [])
        if model_results:
            confidences = [r['confidence'] for r in model_results]
            
            fig = go.Figure(data=go.Histogram(
                x=confidences,
                nbinsx=10,
                marker_color='rgba(55, 128, 191, 0.7)'
            ))
            
            fig.update_layout(
                title="Confidence Score Distribution",
                xaxis_title="Confidence Score",
                yaxis_title="Frequency"
            )
            
            st.plotly_chart(fig, width='stretch')
    
    def render_details_tab(self, results: Dict):
        """Render detailed analysis results."""
        predictions = results.get('ai_predictions', {})
        analysis_details = results.get('analysis_details', {})
        
        # Model Results Table
        overall_analysis = results.get('overall_analysis', {})
        model_results = overall_analysis.get('model_results', [])
        
        if model_results:
            st.subheader("Model Results")
            df = pd.DataFrame(model_results)
            st.dataframe(df, width='stretch')
        
        # Technical Details (in expanders)
        if 'frequency_analysis' in analysis_details:
            with st.expander("Frequency Analysis"):
                freq_analysis = analysis_details['frequency_analysis']
                features = freq_analysis.get('frequency_features', {})
                artifacts = freq_analysis.get('artifacts_detected', {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Features:**")
                    for key, value in features.items():
                        if isinstance(value, (int, float)):
                            st.write(f"- {key.replace('_', ' ').title()}: {value:.4f}")
                
                with col2:
                    st.write("**Artifacts:**")
                    st.write(f"- Anomaly Score: {artifacts.get('anomaly_score', 0):.3f}")
                    st.write(f"- Is Anomalous: {artifacts.get('is_anomalous', False)}")
        
        if 'face_analysis' in analysis_details:
            with st.expander("Face Analysis"):
                face_analysis = analysis_details['face_analysis']
                st.write(f"**Faces Detected:** {face_analysis.get('faces_detected', 0)}")
                quality = face_analysis.get('quality_metrics', {})
                st.write(f"**Detection Confidence:** {quality.get('avg_detection_confidence', 0):.3f}")
        
        # Individual Model Details
        with st.expander("Individual Model Results"):
            for model_name, pred in predictions.items():
                if model_name == 'ensemble' or 'error' in pred:
                    continue
                
                st.write(f"**{model_name.title()}:**")
                if model_name in ['efficientnet', 'mobilenet']:
                    st.write(f"- Confidence: {pred.get('confidence', 0):.3f}")
                    st.write(f"- Prediction: {'FAKE' if pred.get('is_fake', False) else 'REAL'}")
                elif model_name == 'frequency':
                    artifacts = pred.get('artifacts_detected', {})
                    st.write(f"- Anomaly Score: {artifacts.get('anomaly_score', 0):.3f}")
                elif model_name == 'face_analysis':
                    st.write(f"- Faces: {pred.get('faces_detected', 0)}")
                st.write("---")
    
    def render_video_results(self, results: Dict):
        """Render video analysis results with frame-by-frame breakdown."""
        if 'error' in results:
            st.error(f"Error: {results['error']}")
            return
        
        video_analysis = results.get('video_analysis', {})
        video_info = results.get('video_info', {})
        frame_results = results.get('frame_results', [])
        
        # Main video detection result
        overall_confidence = video_analysis.get('overall_confidence', 0.5)
        is_deepfake = video_analysis.get('is_deepfake', False)
        suspicious_ratio = video_analysis.get('suspicious_ratio', 0.0)
        
        if is_deepfake:
            st.error(f"DEEPFAKE DETECTED IN VIDEO - Confidence: {overall_confidence:.1%}")
            st.warning(f"{suspicious_ratio:.1%} of analyzed frames contain suspicious content")
        else:
            st.success(f"VIDEO APPEARS AUTHENTIC - Confidence: {(1-overall_confidence):.1%}")
        
        # Video information
        st.subheader("Video Analysis Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Frames", f"{video_info.get('total_frames', 0):,}")
        with col2:
            st.metric("Processed Frames", video_info.get('frames_processed', 0))
        with col3:
            st.metric("Suspicious Frames", video_analysis.get('suspicious_frames', 0))
        with col4:
            st.metric("Video FPS", f"{video_info.get('fps', 0):.1f}")
        
        # Confidence statistics
        st.subheader("Confidence Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Confidence", f"{video_analysis.get('avg_confidence', 0):.1%}")
        with col2:
            st.metric("Max Confidence", f"{video_analysis.get('max_confidence', 0):.1%}")
        with col3:
            st.metric("Min Confidence", f"{video_analysis.get('min_confidence', 0):.1%}")
        
        # Confidence timeline chart
        if frame_results:
            timestamps = [frame['timestamp'] for frame in frame_results]
            confidences = [frame['analysis'].get('overall_analysis', {}).get('overall_confidence', 0.5) for frame in frame_results]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=confidences,
                mode='lines+markers',
                name='Deepfake Confidence',
                line=dict(color='red', width=2),
                marker=dict(size=4)
            ))
            
            # Add threshold line
            threshold = video_analysis.get('confidence_threshold', 0.5)
            fig.add_hline(y=threshold, line_dash="dash", line_color="orange", 
                         annotation_text=f"Threshold ({threshold:.1%})")
            
            fig.update_layout(
                title="Deepfake Confidence Over Time",
                xaxis_title="Time (seconds)",
                yaxis_title="Deepfake Confidence",
                yaxis=dict(range=[0, 1]),
                height=400
            )
            
            st.plotly_chart(fig, width='stretch')
        
        # Frame-by-frame details
        with st.expander("Frame-by-Frame Analysis"):
            if frame_results:
                # Create dataframe for frame results
                frame_data = []
                for frame in frame_results:
                    analysis = frame['analysis'].get('overall_analysis', {})
                    frame_data.append({
                        'Frame': frame['frame_number'],
                        'Time (s)': f"{frame['timestamp']:.2f}",
                        'Confidence': f"{analysis.get('overall_confidence', 0):.1%}",
                        'Is Deepfake': "Yes" if analysis.get('is_deepfake', False) else "No",
                        'Models Used': analysis.get('total_models_used', 0)
                    })
                
                df = pd.DataFrame(frame_data)
                st.dataframe(df, width='stretch')
            else:
                st.write("No frame analysis data available")
    
    def run(self):
        """Main application runner."""
        self.render_header()
        
        # Sidebar settings
        settings = self.render_sidebar()
        
        # Main content
        st.markdown('<div class="section-header">Upload Content for AI Analysis</div>', unsafe_allow_html=True)
        
        # Dynamic file upload based on media type
        if settings['media_type'] == 'Image':
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                help="Upload an image to analyze with our AI deep fake detection models"
            )
            
            if uploaded_file is not None:
                # Get file data once to avoid stream consumption issues
                file_data = uploaded_file.getvalue()
                
                # Display uploaded image
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(file_data, caption="Uploaded Image", width='stretch')
                
                with col2:
                    with st.expander("File Details"):
                        st.write(f"**Name:** {uploaded_file.name}")
                        st.write(f"**Size:** {len(file_data):,} bytes")
                        st.write(f"**Type:** {uploaded_file.type}")
                
                # Analysis button
                if st.button("Analyze with AI Models", type="primary"):
                    with st.spinner("Running AI deep fake detection models..."):
                        results = self.process_image_with_ai_bytes(file_data, uploaded_file.name, settings)
                        self.render_ai_results(results, "image")
            else:
                st.info("Upload an image to start AI-powered deep fake detection")
                
        else:  # Video mode
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
                help="Upload a video to analyze for deep fake content frame by frame"
            )
            
            if uploaded_file is not None:
                file_data = uploaded_file.getvalue()
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.video(file_data)
                
                with col2:
                    with st.expander("Video Details"):
                        st.write(f"**Name:** {uploaded_file.name}")
                        st.write(f"**Size:** {len(file_data):,} bytes")
                        st.write(f"**Type:** {uploaded_file.type}")
                        st.write(f"**Frame Skip:** Every {settings['frame_skip']} frames")
                        st.write(f"**Max Frames:** {settings['max_frames']} frames")
                
                # Analysis button
                if st.button("Analyze Video with AI Models", type="primary"):
                    with st.spinner(f"Analyzing video frames with AI models... (Processing up to {settings['max_frames']} frames)"):
                        results = self.process_video_with_ai(file_data, uploaded_file.name, settings)
                        self.render_video_results(results)
            else:
                st.info("Upload a video to start AI-powered deep fake detection")
                st.write(f"**Current Settings:** Process every {settings['frame_skip']} frames, max {settings['max_frames']} frames")
            
            with st.expander("About This System"):
                st.write("""
                This system uses multiple AI models including EfficientNet, MobileNet, frequency analysis, 
                and facial landmark detection to identify artificially generated content with high accuracy.
                """)

# Initialize and run the application
if __name__ == "__main__":
    logger.info("Starting Genuine Deep Fake Detection System application")
    detection_system = GenuineDeepFakeDetectionSystem()
    detection_system.run()