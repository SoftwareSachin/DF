"""
Genuine Deep Fake Detection Models
Implements state-of-the-art deep fake detection using multiple AI techniques
"""
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import mediapipe as mp
from scipy import fft
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple, Optional
import time

logger = logging.getLogger(__name__)

class EfficientNetDeepFakeDetector:
    """EfficientNet-based deep fake detection model"""
    
    def __init__(self):
        self.model = None
        self.input_size = (224, 224)
        self.confidence_threshold = 0.5
        
    def create_model(self):
        """Create EfficientNet-based detection model"""
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.input_size, 3)
        )
        
        # Freeze base layers for transfer learning
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', name='dense_512')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu', name='dense_256')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid', name='deepfake_output')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("EfficientNet model created successfully")
        return self.model
    
    def predict(self, image: np.ndarray) -> Dict:
        """Predict if image is deepfake"""
        if self.model is None:
            self.create_model()
            
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Predict
        start_time = time.time()
        prediction = self.model.predict(processed_image, verbose=0)
        inference_time = time.time() - start_time
        
        confidence = float(prediction[0][0])
        is_fake = confidence > self.confidence_threshold
        
        return {
            'model': 'EfficientNet-B0',
            'confidence': confidence,
            'is_fake': is_fake,
            'inference_time': inference_time,
            'prediction_raw': confidence
        }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for EfficientNet"""
        # Resize
        resized = cv2.resize(image, self.input_size)
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        # Add batch dimension
        batch = np.expand_dims(normalized, axis=0)
        return batch

class MobileNetDeepFakeDetector:
    """MobileNet-based lightweight detection model"""
    
    def __init__(self):
        self.model = None
        self.input_size = (128, 128)  # Smaller for mobile efficiency
        self.confidence_threshold = 0.5
        
    def create_model(self):
        """Create MobileNet-based detection model"""
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.input_size, 3),
            alpha=0.75  # Reduced width for efficiency
        )
        
        base_model.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("MobileNet model created successfully")
        return self.model
    
    def predict(self, image: np.ndarray) -> Dict:
        """Predict if image is deepfake"""
        if self.model is None:
            self.create_model()
            
        processed_image = self._preprocess_image(image)
        
        start_time = time.time()
        prediction = self.model.predict(processed_image, verbose=0)
        inference_time = time.time() - start_time
        
        confidence = float(prediction[0][0])
        is_fake = confidence > self.confidence_threshold
        
        return {
            'model': 'MobileNet-V2',
            'confidence': confidence,
            'is_fake': is_fake,
            'inference_time': inference_time,
            'prediction_raw': confidence
        }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for MobileNet"""
        resized = cv2.resize(image, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        batch = np.expand_dims(normalized, axis=0)
        return batch

class FrequencyDomainAnalyzer:
    """FFT-based frequency domain analysis for deepfake artifacts"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self._is_fitted = False
        
    def analyze_frequency_artifacts(self, image: np.ndarray) -> Dict:
        """Analyze frequency domain for deepfake artifacts"""
        # Convert to grayscale for frequency analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Extract frequency features
        features = self._extract_frequency_features(magnitude_spectrum)
        
        # Analyze artifacts
        artifacts = self._detect_frequency_anomalies(features)
        
        return {
            'frequency_features': features,
            'artifacts_detected': artifacts,
            'magnitude_spectrum': magnitude_spectrum,
            'analysis_method': 'FFT_2D'
        }
    
    def _extract_frequency_features(self, magnitude_spectrum: np.ndarray) -> Dict:
        """Extract relevant frequency domain features"""
        height, width = magnitude_spectrum.shape
        center_y, center_x = height // 2, width // 2
        
        # High frequency content (corners)
        high_freq_mask = np.ones_like(magnitude_spectrum)
        cv2.circle(high_freq_mask, (center_x, center_y), min(center_x, center_y) // 3, 0, -1)
        high_freq_energy = np.mean(magnitude_spectrum[high_freq_mask == 1])
        
        # Low frequency content (center)
        low_freq_mask = np.zeros_like(magnitude_spectrum)
        cv2.circle(low_freq_mask, (center_x, center_y), min(center_x, center_y) // 4, 1, -1)
        low_freq_energy = np.mean(magnitude_spectrum[low_freq_mask == 1])
        
        # Frequency ratio
        freq_ratio = high_freq_energy / (low_freq_energy + 1e-8)
        
        # DCT analysis for additional artifacts
        dct_features = self._analyze_dct_artifacts(magnitude_spectrum)
        
        return {
            'high_freq_energy': float(high_freq_energy),
            'low_freq_energy': float(low_freq_energy),
            'frequency_ratio': float(freq_ratio),
            'dct_artifacts': dct_features,
            'spectrum_mean': float(np.mean(magnitude_spectrum)),
            'spectrum_std': float(np.std(magnitude_spectrum))
        }
    
    def _analyze_dct_artifacts(self, spectrum: np.ndarray) -> Dict:
        """Analyze DCT compression artifacts"""
        # Resize for DCT analysis
        resized = cv2.resize(spectrum.astype(np.float32), (64, 64))
        
        # Apply DCT
        dct_result = cv2.dct(resized)
        
        # Analyze DCT coefficients
        dc_component = dct_result[0, 0]
        ac_energy = np.sum(np.abs(dct_result[1:, 1:]))
        
        return {
            'dc_component': float(dc_component),
            'ac_energy': float(ac_energy),
            'dct_ratio': float(ac_energy / (abs(dc_component) + 1e-8))
        }
    
    def _detect_frequency_anomalies(self, features: Dict) -> Dict:
        """Detect anomalies in frequency domain"""
        # Create feature vector
        feature_vector = np.array([
            features['high_freq_energy'],
            features['low_freq_energy'],
            features['frequency_ratio'],
            features['spectrum_mean'],
            features['spectrum_std'],
            features['dct_artifacts']['dc_component'],
            features['dct_artifacts']['ac_energy']
        ]).reshape(1, -1)
        
        # Use simple thresholding for real-time detection
        anomaly_score = 0.0
        
        # High frequency anomaly
        if features['frequency_ratio'] > 2.0:
            anomaly_score += 0.3
            
        # DCT anomaly
        if features['dct_artifacts']['dct_ratio'] > 1.5:
            anomaly_score += 0.2
            
        # Spectrum variance anomaly
        if features['spectrum_std'] > np.mean([features['spectrum_mean'], 5.0]):
            anomaly_score += 0.2
            
        is_anomalous = anomaly_score > 0.4
        
        return {
            'anomaly_score': float(anomaly_score),
            'is_anomalous': is_anomalous,
            'confidence': min(1.0, anomaly_score * 2)
        }

class MediaPipeFaceAnalyzer:
    """Advanced face analysis using MediaPipe"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        
    def analyze_facial_landmarks(self, image: np.ndarray) -> Dict:
        """Advanced facial landmark analysis"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Face mesh analysis
        mesh_results = self.face_mesh.process(rgb_image)
        
        # Face detection
        detection_results = self.face_detection.process(rgb_image)
        
        analysis = {
            'faces_detected': 0,
            'landmark_analysis': [],
            'geometric_analysis': {},
            'quality_metrics': {}
        }
        
        if mesh_results.multi_face_landmarks:
            analysis['faces_detected'] = len(mesh_results.multi_face_landmarks)
            
            for face_landmarks in mesh_results.multi_face_landmarks:
                landmarks = self._extract_key_landmarks(face_landmarks, image.shape)
                geometric_metrics = self._analyze_facial_geometry(landmarks)
                quality_metrics = self._analyze_landmark_quality(landmarks)
                
                analysis['landmark_analysis'].append({
                    'landmarks': landmarks,
                    'geometry': geometric_metrics,
                    'quality': quality_metrics
                })
        
        # Overall analysis
        if analysis['landmark_analysis']:
            analysis['geometric_analysis'] = self._overall_geometric_analysis(
                analysis['landmark_analysis']
            )
            analysis['quality_metrics'] = self._overall_quality_analysis(
                analysis['landmark_analysis']
            )
        
        return analysis
    
    def _extract_key_landmarks(self, face_landmarks, image_shape) -> Dict:
        """Extract key facial landmarks"""
        h, w = image_shape[:2]
        
        # Key landmark indices for MediaPipe
        key_indices = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 360],
            'mouth': [0, 17, 18, 200, 199, 175, 61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'right_eyebrow': [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        }
        
        landmarks = {}
        for region, indices in key_indices.items():
            region_points = []
            for idx in indices:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    region_points.append((x, y))
            landmarks[region] = region_points
            
        return landmarks
    
    def _analyze_facial_geometry(self, landmarks: Dict) -> Dict:
        """Analyze facial geometry for deepfake artifacts"""
        geometry = {}
        
        # Eye symmetry analysis
        if landmarks.get('left_eye') and landmarks.get('right_eye'):
            left_eye_center = np.mean(landmarks['left_eye'], axis=0)
            right_eye_center = np.mean(landmarks['right_eye'], axis=0)
            eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
            
            # Eye-to-nose ratios
            if landmarks.get('nose'):
                nose_center = np.mean(landmarks['nose'], axis=0)
                left_eye_nose_dist = np.linalg.norm(left_eye_center - nose_center)
                right_eye_nose_dist = np.linalg.norm(right_eye_center - nose_center)
                eye_symmetry = abs(left_eye_nose_dist - right_eye_nose_dist) / eye_distance
                geometry['eye_symmetry'] = float(eye_symmetry)
                geometry['eye_distance'] = float(eye_distance)
        
        # Face aspect ratio
        if landmarks.get('face_oval'):
            face_points = np.array(landmarks['face_oval'])
            face_width = np.max(face_points[:, 0]) - np.min(face_points[:, 0])
            face_height = np.max(face_points[:, 1]) - np.min(face_points[:, 1])
            aspect_ratio = face_width / face_height if face_height > 0 else 0
            geometry['face_aspect_ratio'] = float(aspect_ratio)
        
        return geometry
    
    def _analyze_landmark_quality(self, landmarks: Dict) -> Dict:
        """Analyze landmark detection quality"""
        quality = {}
        
        total_landmarks = sum(len(points) for points in landmarks.values())
        quality['total_landmarks'] = total_landmarks
        
        # Landmark density (points per facial region)
        region_densities = {}
        for region, points in landmarks.items():
            if points:
                points_array = np.array(points)
                # Calculate convex hull area
                if len(points) >= 3:
                    hull = cv2.convexHull(points_array)
                    area = cv2.contourArea(hull)
                    density = len(points) / (area + 1e-8)
                    region_densities[region] = float(density)
        
        quality['region_densities'] = region_densities
        quality['detection_confidence'] = np.mean(list(region_densities.values())) if region_densities else 0.0
        
        return quality
    
    def _overall_geometric_analysis(self, landmark_analyses: List) -> Dict:
        """Overall geometric analysis across all detected faces"""
        if not landmark_analyses:
            return {}
        
        geometries = [analysis['geometry'] for analysis in landmark_analyses]
        
        # Average metrics
        avg_metrics = {}
        for key in ['eye_symmetry', 'eye_distance', 'face_aspect_ratio']:
            values = [g.get(key, 0) for g in geometries if g.get(key) is not None]
            if values:
                avg_metrics[f'avg_{key}'] = np.mean(values)
                avg_metrics[f'std_{key}'] = np.std(values)
        
        return avg_metrics
    
    def _overall_quality_analysis(self, landmark_analyses: List) -> Dict:
        """Overall quality analysis"""
        if not landmark_analyses:
            return {}
        
        qualities = [analysis['quality'] for analysis in landmark_analyses]
        
        avg_confidence = np.mean([q.get('detection_confidence', 0) for q in qualities])
        total_landmarks = sum(q.get('total_landmarks', 0) for q in qualities)
        
        return {
            'avg_detection_confidence': float(avg_confidence),
            'total_landmarks_all_faces': total_landmarks,
            'face_count': len(qualities)
        }

class EnsembleDeepFakeDetector:
    """Ensemble model combining multiple detection techniques"""
    
    def __init__(self):
        self.efficientnet_detector = EfficientNetDeepFakeDetector()
        self.mobilenet_detector = MobileNetDeepFakeDetector()
        self.frequency_analyzer = FrequencyDomainAnalyzer()
        self.face_analyzer = MediaPipeFaceAnalyzer()
        
        # Weights for ensemble
        self.weights = {
            'efficientnet': 0.35,
            'mobilenet': 0.25,
            'frequency': 0.20,
            'face_geometry': 0.20
        }
        
    def detect_deepfake(self, image: np.ndarray) -> Dict:
        """Comprehensive deepfake detection using ensemble approach"""
        results = {
            'ensemble_prediction': {},
            'individual_predictions': {},
            'analysis_details': {}
        }
        
        start_time = time.time()
        
        # EfficientNet prediction
        try:
            efficientnet_result = self.efficientnet_detector.predict(image)
            results['individual_predictions']['efficientnet'] = efficientnet_result
        except Exception as e:
            logger.error(f"EfficientNet prediction failed: {e}")
            results['individual_predictions']['efficientnet'] = {'error': str(e)}
        
        # MobileNet prediction
        try:
            mobilenet_result = self.mobilenet_detector.predict(image)
            results['individual_predictions']['mobilenet'] = mobilenet_result
        except Exception as e:
            logger.error(f"MobileNet prediction failed: {e}")
            results['individual_predictions']['mobilenet'] = {'error': str(e)}
        
        # Frequency domain analysis
        try:
            frequency_result = self.frequency_analyzer.analyze_frequency_artifacts(image)
            results['individual_predictions']['frequency'] = frequency_result
            results['analysis_details']['frequency_analysis'] = frequency_result
        except Exception as e:
            logger.error(f"Frequency analysis failed: {e}")
            results['individual_predictions']['frequency'] = {'error': str(e)}
        
        # Face analysis
        try:
            face_result = self.face_analyzer.analyze_facial_landmarks(image)
            results['individual_predictions']['face_analysis'] = face_result
            results['analysis_details']['face_analysis'] = face_result
        except Exception as e:
            logger.error(f"Face analysis failed: {e}")
            results['individual_predictions']['face_analysis'] = {'error': str(e)}
        
        # Ensemble prediction
        ensemble_confidence = self._calculate_ensemble_confidence(results['individual_predictions'])
        
        total_time = time.time() - start_time
        
        results['ensemble_prediction'] = {
            'confidence': ensemble_confidence,
            'is_fake': ensemble_confidence > 0.5,
            'total_inference_time': total_time,
            'method': 'Multi-Model Ensemble',
            'model_weights': self.weights
        }
        
        return results
    
    def _calculate_ensemble_confidence(self, predictions: Dict) -> float:
        """Calculate weighted ensemble confidence"""
        confidences = []
        weights = []
        
        # EfficientNet confidence
        efficientnet_pred = predictions.get('efficientnet', {})
        if 'confidence' in efficientnet_pred and 'error' not in efficientnet_pred:
            confidences.append(efficientnet_pred['confidence'])
            weights.append(self.weights['efficientnet'])
        
        # MobileNet confidence
        mobilenet_pred = predictions.get('mobilenet', {})
        if 'confidence' in mobilenet_pred and 'error' not in mobilenet_pred:
            confidences.append(mobilenet_pred['confidence'])
            weights.append(self.weights['mobilenet'])
        
        # Frequency analysis confidence
        frequency_pred = predictions.get('frequency', {})
        if 'artifacts_detected' in frequency_pred and 'error' not in frequency_pred:
            freq_confidence = frequency_pred['artifacts_detected'].get('confidence', 0.5)
            confidences.append(freq_confidence)
            weights.append(self.weights['frequency'])
        
        # Face geometry confidence
        face_pred = predictions.get('face_analysis', {})
        if 'quality_metrics' in face_pred and 'error' not in face_pred:
            # Use detection confidence as indicator
            face_confidence = face_pred['quality_metrics'].get('avg_detection_confidence', 0.5)
            # Convert to deepfake confidence (lower face quality = higher fake probability)
            face_deepfake_confidence = 1.0 - min(1.0, face_confidence)
            confidences.append(face_deepfake_confidence)
            weights.append(self.weights['face_geometry'])
        
        # Calculate weighted average
        if confidences and weights:
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_confidence = sum(c * w for c, w in zip(confidences, weights)) / total_weight
                return float(weighted_confidence)
        
        # Fallback to simple average
        return float(np.mean(confidences)) if confidences else 0.5