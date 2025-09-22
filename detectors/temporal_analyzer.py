"""
Temporal analysis for video deep fake detection.
Analyzes frame-to-frame inconsistencies and temporal artifacts.
"""
import numpy as np
import cv2
from typing import List, Dict, Tuple

# Optional imports with fallbacks
try:
    from scipy import signal
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some temporal analysis features will be limited.")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting features disabled.")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Using fallback similarity calculations.")
    
    def cosine_similarity(X, Y):
        """Simple cosine similarity fallback"""
        if len(X) == 0 or len(Y) == 0:
            return np.array([[0.0]])
        
        X_norm = X / np.linalg.norm(X)
        Y_norm = Y / np.linalg.norm(Y)
        return np.array([[np.dot(X_norm.flatten(), Y_norm.flatten())]])

class TemporalAnalyzer:
    """
    Analyzes temporal inconsistencies in video sequences for deep fake detection.
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.optical_flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    
    def analyze_temporal_consistency(self, frames: List[np.ndarray], face_bboxes: List[Tuple[int, int, int, int]]) -> Dict:
        """
        Analyze temporal consistency across video frames.
        
        Args:
            frames: List of video frames
            face_bboxes: List of face bounding boxes for each frame
            
        Returns:
            Dictionary of temporal analysis results
        """
        if len(frames) < 2:
            return {'error': 'Insufficient frames for temporal analysis'}
        
        analysis = {}
        
        # Optical flow analysis
        analysis['optical_flow'] = self._analyze_optical_flow(frames, face_bboxes)
        
        # Frame difference analysis
        analysis['frame_differences'] = self._analyze_frame_differences(frames, face_bboxes)
        
        # Facial feature tracking
        analysis['feature_tracking'] = self._analyze_feature_tracking(frames, face_bboxes)
        
        # Illumination consistency
        analysis['illumination'] = self._analyze_illumination_consistency(frames, face_bboxes)
        
        # Texture consistency
        analysis['texture'] = self._analyze_texture_consistency(frames, face_bboxes)
        
        # Motion smoothness
        analysis['motion_smoothness'] = self._analyze_motion_smoothness(frames, face_bboxes)
        
        # Temporal frequency analysis
        analysis['frequency_analysis'] = self._analyze_temporal_frequencies(frames, face_bboxes)
        
        return analysis
    
    def _analyze_optical_flow(self, frames: List[np.ndarray], face_bboxes: List[Tuple[int, int, int, int]]) -> Dict:
        """Analyze optical flow patterns in facial regions."""
        flow_analysis = {
            'flow_magnitudes': [],
            'flow_directions': [],
            'flow_consistency': [],
            'anomalous_motion': 0
        }
        
        prev_frame = None
        prev_bbox = None
        
        for i, (frame, bbox) in enumerate(zip(frames, face_bboxes)):
            if prev_frame is not None and prev_bbox is not None:
                # Extract face regions
                x1, y1, w1, h1 = prev_bbox
                x2, y2, w2, h2 = bbox
                
                # Convert to grayscale
                gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)[y1:y1+h1, x1:x1+w1]
                gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y2:y2+h2, x2:x2+w2]
                
                # Resize to same size if different
                if gray1.shape != gray2.shape:
                    min_h, min_w = min(gray1.shape[0], gray2.shape[0]), min(gray1.shape[1], gray2.shape[1])
                    gray1 = cv2.resize(gray1, (min_w, min_h))
                    gray2 = cv2.resize(gray2, (min_w, min_h))
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    gray1, gray2,
                    np.array([[gray1.shape[1]//2, gray1.shape[0]//2]], dtype=np.float32),
                    None,
                    **self.optical_flow_params
                )[0]
                
                if flow is not None and len(flow) > 0:
                    # Calculate flow magnitude and direction
                    magnitude = np.sqrt(flow[0][0]**2 + flow[0][1]**2)
                    direction = np.arctan2(flow[0][1], flow[0][0])
                    
                    flow_analysis['flow_magnitudes'].append(float(magnitude))
                    flow_analysis['flow_directions'].append(float(direction))
                    
                    # Check for anomalous motion
                    if magnitude > 50:  # Threshold for anomalous motion
                        flow_analysis['anomalous_motion'] += 1
            
            prev_frame = frame
            prev_bbox = bbox
        
        # Calculate flow consistency
        if len(flow_analysis['flow_magnitudes']) > 1:
            magnitudes = np.array(flow_analysis['flow_magnitudes'])
            directions = np.array(flow_analysis['flow_directions'])
            
            flow_analysis['magnitude_variance'] = float(np.var(magnitudes))
            flow_analysis['direction_variance'] = float(np.var(directions))
            flow_analysis['flow_consistency'] = float(1.0 / (1.0 + flow_analysis['magnitude_variance']))
        
        return flow_analysis
    
    def _analyze_frame_differences(self, frames: List[np.ndarray], face_bboxes: List[Tuple[int, int, int, int]]) -> Dict:
        """Analyze frame-to-frame differences in facial regions."""
        differences = {
            'mean_differences': [],
            'edge_differences': [],
            'color_differences': [],
            'difference_consistency': 0.0
        }
        
        for i in range(1, len(frames)):
            prev_frame = frames[i-1]
            curr_frame = frames[i]
            prev_bbox = face_bboxes[i-1]
            curr_bbox = face_bboxes[i]
            
            # Extract face regions
            x1, y1, w1, h1 = prev_bbox
            x2, y2, w2, h2 = curr_bbox
            
            face1 = prev_frame[y1:y1+h1, x1:x1+w1]
            face2 = curr_frame[y2:y2+h2, x2:x2+w2]
            
            # Resize to same size
            if face1.shape != face2.shape:
                min_h, min_w = min(face1.shape[0], face2.shape[0]), min(face1.shape[1], face2.shape[1])
                face1 = cv2.resize(face1, (min_w, min_h))
                face2 = cv2.resize(face2, (min_w, min_h))
            
            # Calculate mean absolute difference
            mean_diff = np.mean(np.abs(face1.astype(float) - face2.astype(float)))
            differences['mean_differences'].append(float(mean_diff))
            
            # Calculate edge differences
            gray1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
            
            edges1 = cv2.Canny(gray1, 50, 150)
            edges2 = cv2.Canny(gray2, 50, 150)
            
            edge_diff = np.mean(np.abs(edges1.astype(float) - edges2.astype(float)))
            differences['edge_differences'].append(float(edge_diff))
            
            # Calculate color channel differences
            color_diff = np.mean(np.abs(face1 - face2), axis=(0, 1))
            differences['color_differences'].append(color_diff.tolist())
        
        # Calculate consistency metrics
        if len(differences['mean_differences']) > 0:
            differences['difference_consistency'] = float(1.0 / (1.0 + np.var(differences['mean_differences'])))
        
        return differences
    
    def _analyze_feature_tracking(self, frames: List[np.ndarray], face_bboxes: List[Tuple[int, int, int, int]]) -> Dict:
        """Analyze facial feature tracking consistency."""
        tracking_analysis = {
            'feature_points': [],
            'tracking_quality': [],
            'lost_tracks': 0,
            'tracking_stability': 0.0
        }
        
        # Initialize feature points from first frame
        if len(frames) > 0 and len(face_bboxes) > 0:
            x, y, w, h = face_bboxes[0]
            face_region = frames[0][y:y+h, x:x+w]
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Detect corners for tracking
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=100,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7
            )
            
            if corners is not None:
                prev_gray = gray
                prev_points = corners
                
                for i in range(1, len(frames)):
                    x, y, w, h = face_bboxes[i]
                    face_region = frames[i][y:y+h, x:x+w]
                    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                    
                    # Track features
                    next_points, status, error = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray, prev_points, None,
                        **self.optical_flow_params
                    )
                    
                    # Filter good points
                    good_new = next_points[status == 1]
                    good_old = prev_points[status == 1]
                    
                    if len(good_new) > 0:
                        # Calculate tracking quality
                        distances = np.sqrt(np.sum((good_new - good_old)**2, axis=1))
                        tracking_quality = 1.0 / (1.0 + np.mean(distances))
                        tracking_analysis['tracking_quality'].append(float(tracking_quality))
                        
                        # Count lost tracks
                        lost_count = len(prev_points) - len(good_new)
                        tracking_analysis['lost_tracks'] += lost_count
                        
                        prev_points = good_new.reshape(-1, 1, 2)
                    else:
                        tracking_analysis['lost_tracks'] += len(prev_points)
                        break
                    
                    prev_gray = gray
        
        # Calculate overall tracking stability
        if len(tracking_analysis['tracking_quality']) > 0:
            tracking_analysis['tracking_stability'] = float(np.mean(tracking_analysis['tracking_quality']))
        
        return tracking_analysis
    
    def _analyze_illumination_consistency(self, frames: List[np.ndarray], face_bboxes: List[Tuple[int, int, int, int]]) -> Dict:
        """Analyze illumination consistency across frames."""
        illumination_analysis = {
            'brightness_values': [],
            'brightness_variance': 0.0,
            'illumination_jumps': 0,
            'gradient_consistency': 0.0
        }
        
        brightness_history = []
        
        for frame, bbox in zip(frames, face_bboxes):
            x, y, w, h = bbox
            face_region = frame[y:y+h, x:x+w]
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate average brightness
            brightness = np.mean(gray_face)
            brightness_history.append(brightness)
            
            # Calculate brightness gradient
            grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            illumination_analysis['brightness_values'].append(float(brightness))
        
        # Analyze brightness consistency
        if len(brightness_history) > 1:
            brightness_array = np.array(brightness_history)
            illumination_analysis['brightness_variance'] = float(np.var(brightness_array))
            
            # Count sudden illumination jumps
            brightness_diff = np.abs(np.diff(brightness_array))
            jump_threshold = np.std(brightness_diff) * 2
            illumination_analysis['illumination_jumps'] = int(np.sum(brightness_diff > jump_threshold))
            
            # Calculate gradient consistency
            illumination_analysis['gradient_consistency'] = float(1.0 / (1.0 + illumination_analysis['brightness_variance']))
        
        return illumination_analysis
    
    def _analyze_texture_consistency(self, frames: List[np.ndarray], face_bboxes: List[Tuple[int, int, int, int]]) -> Dict:
        """Analyze texture consistency across frames."""
        texture_analysis = {
            'texture_features': [],
            'texture_variance': 0.0,
            'texture_correlation': 0.0
        }
        
        texture_features = []
        
        for frame, bbox in zip(frames, face_bboxes):
            x, y, w, h = bbox
            face_region = frame[y:y+h, x:x+w]
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate Local Binary Pattern features
            lbp_features = self._calculate_lbp_features(gray_face)
            texture_features.append(lbp_features)
        
        if len(texture_features) > 1:
            texture_matrix = np.array(texture_features)
            
            # Calculate texture variance
            texture_analysis['texture_variance'] = float(np.mean(np.var(texture_matrix, axis=0)))
            
            # Calculate temporal correlation
            correlations = []
            for i in range(1, len(texture_features)):
                corr = cosine_similarity([texture_features[i-1]], [texture_features[i]])[0][0]
                correlations.append(corr)
            
            if correlations:
                texture_analysis['texture_correlation'] = float(np.mean(correlations))
        
        return texture_analysis
    
    def _calculate_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern features."""
        # Simple LBP implementation
        height, width = image.shape
        lbp_image = np.zeros((height-2, width-2), dtype=np.uint8)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = image[i, j]
                code = 0
                
                # 8-neighborhood
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp_image[i-1, j-1] = code
        
        # Calculate histogram
        hist, _ = np.histogram(lbp_image.flatten(), bins=256, range=(0, 256))
        return hist.astype(np.float32) / np.sum(hist)
    
    def _analyze_motion_smoothness(self, frames: List[np.ndarray], face_bboxes: List[Tuple[int, int, int, int]]) -> Dict:
        """Analyze motion smoothness of facial movement."""
        motion_analysis = {
            'position_changes': [],
            'velocity_changes': [],
            'acceleration_changes': [],
            'motion_smoothness_score': 0.0
        }
        
        positions = []
        
        for bbox in face_bboxes:
            x, y, w, h = bbox
            center_x, center_y = x + w/2, y + h/2
            positions.append([center_x, center_y])
        
        if len(positions) > 2:
            positions = np.array(positions)
            
            # Calculate velocities
            velocities = np.diff(positions, axis=0)
            motion_analysis['position_changes'] = velocities.tolist()
            
            # Calculate accelerations
            accelerations = np.diff(velocities, axis=0)
            motion_analysis['acceleration_changes'] = accelerations.tolist()
            
            # Calculate smoothness score
            velocity_magnitude = np.sqrt(np.sum(velocities**2, axis=1))
            acceleration_magnitude = np.sqrt(np.sum(accelerations**2, axis=1))
            
            if len(acceleration_magnitude) > 0:
                smoothness = 1.0 / (1.0 + np.mean(acceleration_magnitude))
                motion_analysis['motion_smoothness_score'] = float(smoothness)
        
        return motion_analysis
    
    def _analyze_temporal_frequencies(self, frames: List[np.ndarray], face_bboxes: List[Tuple[int, int, int, int]]) -> Dict:
        """Analyze temporal frequency patterns."""
        frequency_analysis = {
            'dominant_frequencies': [],
            'frequency_stability': 0.0,
            'anomalous_frequencies': 0
        }
        
        # Extract pixel intensity time series
        pixel_series = []
        
        for frame, bbox in zip(frames, face_bboxes):
            x, y, w, h = bbox
            face_region = frame[y:y+h, x:x+w]
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Sample key pixels
            center_intensity = gray_face[h//2, w//2]
            pixel_series.append(center_intensity)
        
        if len(pixel_series) > 10:
            # Perform FFT analysis
            fft_result = np.fft.fft(pixel_series)
            frequencies = np.fft.fftfreq(len(pixel_series))
            
            # Find dominant frequencies
            magnitude = np.abs(fft_result)
            dominant_indices = np.argsort(magnitude)[-5:]  # Top 5 frequencies
            
            for idx in dominant_indices:
                frequency_analysis['dominant_frequencies'].append(float(frequencies[idx]))
            
            # Calculate frequency stability
            frequency_variance = np.var(magnitude)
            frequency_analysis['frequency_stability'] = float(1.0 / (1.0 + frequency_variance))
            
            # Detect anomalous high frequencies
            high_freq_threshold = 0.3  # Normalized frequency
            anomalous_count = np.sum(np.abs(frequencies[dominant_indices]) > high_freq_threshold)
            frequency_analysis['anomalous_frequencies'] = int(anomalous_count)
        
        return frequency_analysis
