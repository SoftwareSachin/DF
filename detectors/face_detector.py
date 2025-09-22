"""
Face detection and facial landmark analysis for deep fake detection.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import math

# Optional imports with fallbacks
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib not available. Some facial analysis features will be limited.")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not available. Using OpenCV for face detection.")

try:
    from scipy.spatial.distance import euclidean
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Using basic distance calculations.")
    
    def euclidean(p1, p2):
        """Simple Euclidean distance fallback"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

class FaceDetector:
    """
    Advanced face detection and analysis for deep fake detection.
    """
    
    def __init__(self):
        # Initialize OpenCV face detector as fallback
        self.opencv_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize dlib if available
        if DLIB_AVAILABLE:
            self.face_detector = dlib.get_frontal_face_detector()
            
            # Try to load the landmark predictor (would need to be downloaded)
            try:
                self.landmark_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
                self.dlib_landmarks_available = True
            except:
                print("Warning: dlib landmark predictor not found. Facial landmark analysis will be limited.")
                self.dlib_landmarks_available = False
        else:
            self.face_detector = None
            self.dlib_landmarks_available = False
        
        # Initialize MediaPipe if available
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_face_detection = mp.solutions.face_detection
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )
        else:
            self.mp_face_mesh = None
            self.mp_face_detection = None
            self.face_mesh = None
            self.face_detection = None
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the image.
        
        Args:
            image: Input image
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        faces = []
        
        # Try dlib first if available
        if DLIB_AVAILABLE and self.face_detector is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detected_faces = self.face_detector(gray)
            
            for face in detected_faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                faces.append((x, y, w, h))
                
        # Try MediaPipe if dlib didn't work and MediaPipe is available
        elif MEDIAPIPE_AVAILABLE and self.face_detection is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_image)
            
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    x = int(bboxC.xmin * w)
                    y = int(bboxC.ymin * h)
                    width = int(bboxC.width * w)
                    height = int(bboxC.height * h)
                    faces.append((x, y, width, height))
        
        # Fall back to OpenCV Haar Cascade if nothing else worked
        if not faces:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            opencv_faces = self.opencv_face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            faces = [(x, y, w, h) for (x, y, w, h) in opencv_faces]
        
        return faces
    
    def extract_landmarks(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from a detected face.
        
        Args:
            image: Input image
            face_bbox: Face bounding box (x, y, w, h)
            
        Returns:
            Array of landmark points or None if extraction fails
        """
        x, y, w, h = face_bbox
        
        if self.dlib_available:
            # Use dlib for landmark detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_rect = dlib.rectangle(x, y, x + w, y + h)
            landmarks = self.landmark_predictor(gray, face_rect)
            
            # Convert to numpy array
            points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
            return points
        else:
            # Use MediaPipe for landmark detection
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                h_img, w_img, _ = image.shape
                points = []
                
                # Extract key landmarks (68 point equivalent)
                key_indices = [
                    # Face outline
                    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
                    # Eyebrows
                    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
                    285, 295, 282, 283, 276, 293, 334, 296, 336, 285,
                    # Eyes
                    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
                    159, 160, 161, 246, 33, 130,
                    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
                    386, 385, 384, 398, 362, 398,
                    # Nose
                    19, 20, 240, 131, 134, 102, 48, 64, 98, 240,
                    # Mouth
                    61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
                    402, 317, 14, 87, 178, 88, 95, 78
                ]
                
                for idx in key_indices[:68]:  # Limit to 68 points
                    if idx < len(landmarks.landmark):
                        landmark = landmarks.landmark[idx]
                        points.append([int(landmark.x * w_img), int(landmark.y * h_img)])
                
                return np.array(points) if points else None
        
        return None
    
    def analyze_facial_geometry(self, landmarks: np.ndarray) -> Dict:
        """
        Analyze facial geometry for inconsistencies.
        
        Args:
            landmarks: Array of facial landmark points
            
        Returns:
            Dictionary of geometric analysis results
        """
        if landmarks is None or len(landmarks) < 68:
            return {}
        
        analysis = {}
        
        # Calculate facial symmetry
        analysis['symmetry_score'] = self._calculate_symmetry(landmarks)
        
        # Calculate facial proportions
        analysis['proportions'] = self._calculate_proportions(landmarks)
        
        # Calculate angles and distances
        analysis['geometric_features'] = self._calculate_geometric_features(landmarks)
        
        # Detect geometric anomalies
        analysis['anomalies'] = self._detect_geometric_anomalies(landmarks)
        
        return analysis
    
    def _calculate_symmetry(self, landmarks: np.ndarray) -> float:
        """Calculate facial symmetry score."""
        # Get face center line
        nose_tip = landmarks[30]  # Nose tip
        chin = landmarks[8]       # Chin
        center_x = (nose_tip[0] + chin[0]) / 2
        
        # Calculate symmetry for key facial features
        symmetry_scores = []
        
        # Eye symmetry
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)
        
        left_distance = abs(left_eye_center[0] - center_x)
        right_distance = abs(right_eye_center[0] - center_x)
        
        if left_distance + right_distance > 0:
            eye_symmetry = 1 - abs(left_distance - right_distance) / (left_distance + right_distance)
            symmetry_scores.append(eye_symmetry)
        
        # Mouth corner symmetry
        left_mouth = landmarks[48]
        right_mouth = landmarks[54]
        
        left_mouth_distance = abs(left_mouth[0] - center_x)
        right_mouth_distance = abs(right_mouth[0] - center_x)
        
        if left_mouth_distance + right_mouth_distance > 0:
            mouth_symmetry = 1 - abs(left_mouth_distance - right_mouth_distance) / (left_mouth_distance + right_mouth_distance)
            symmetry_scores.append(mouth_symmetry)
        
        return float(np.mean(symmetry_scores)) if symmetry_scores else 0.0
    
    def _calculate_proportions(self, landmarks: np.ndarray) -> Dict:
        """Calculate facial proportions."""
        proportions = {}
        
        # Face width and height
        face_width = euclidean(landmarks[0], landmarks[16])  # Jaw width
        face_height = euclidean(landmarks[8], landmarks[19])  # Chin to forehead
        
        if face_height > 0:
            proportions['width_height_ratio'] = face_width / face_height
        
        # Eye separation
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)
        eye_separation = euclidean(left_eye_center, right_eye_center)
        
        # Eye width
        left_eye_width = euclidean(landmarks[36], landmarks[39])
        right_eye_width = euclidean(landmarks[42], landmarks[45])
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        
        if avg_eye_width > 0:
            proportions['eye_separation_ratio'] = eye_separation / avg_eye_width
        
        # Nose width to mouth width ratio
        nose_width = euclidean(landmarks[31], landmarks[35])
        mouth_width = euclidean(landmarks[48], landmarks[54])
        
        if mouth_width > 0:
            proportions['nose_mouth_ratio'] = nose_width / mouth_width
        
        return proportions
    
    def _calculate_geometric_features(self, landmarks: np.ndarray) -> Dict:
        """Calculate geometric features and angles."""
        features = {}
        
        # Eye angles
        left_eye_angle = self._calculate_eye_angle(landmarks[36:42])
        right_eye_angle = self._calculate_eye_angle(landmarks[42:48])
        
        features['left_eye_angle'] = left_eye_angle
        features['right_eye_angle'] = right_eye_angle
        features['eye_angle_difference'] = abs(left_eye_angle - right_eye_angle)
        
        # Mouth angle
        mouth_left = landmarks[48]
        mouth_right = landmarks[54]
        mouth_center_top = landmarks[51]
        mouth_center_bottom = landmarks[57]
        
        mouth_angle = math.atan2(mouth_right[1] - mouth_left[1], mouth_right[0] - mouth_left[0])
        features['mouth_angle'] = math.degrees(mouth_angle)
        
        # Nose angle
        nose_tip = landmarks[30]
        nose_bridge = landmarks[27]
        nose_angle = math.atan2(nose_tip[1] - nose_bridge[1], nose_tip[0] - nose_bridge[0])
        features['nose_angle'] = math.degrees(nose_angle)
        
        return features
    
    def _calculate_eye_angle(self, eye_landmarks: np.ndarray) -> float:
        """Calculate the angle of an eye."""
        if len(eye_landmarks) < 6:
            return 0.0
        
        left_corner = eye_landmarks[0]
        right_corner = eye_landmarks[3]
        
        angle = math.atan2(right_corner[1] - left_corner[1], right_corner[0] - left_corner[0])
        return math.degrees(angle)
    
    def _detect_geometric_anomalies(self, landmarks: np.ndarray) -> List[str]:
        """Detect geometric anomalies that might indicate manipulation."""
        anomalies = []
        
        # Check for extreme asymmetry
        symmetry_score = self._calculate_symmetry(landmarks)
        if symmetry_score < 0.7:
            anomalies.append("facial_asymmetry")
        
        # Check for unusual proportions
        proportions = self._calculate_proportions(landmarks)
        
        if 'width_height_ratio' in proportions:
            if proportions['width_height_ratio'] < 0.6 or proportions['width_height_ratio'] > 1.2:
                anomalies.append("unusual_face_proportions")
        
        if 'eye_separation_ratio' in proportions:
            if proportions['eye_separation_ratio'] < 2.0 or proportions['eye_separation_ratio'] > 4.0:
                anomalies.append("unusual_eye_spacing")
        
        # Check for geometric inconsistencies
        features = self._calculate_geometric_features(landmarks)
        
        if features.get('eye_angle_difference', 0) > 10:
            anomalies.append("eye_angle_inconsistency")
        
        if abs(features.get('mouth_angle', 0)) > 15:
            anomalies.append("unusual_mouth_angle")
        
        return anomalies
    
    def analyze_face_quality(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Analyze face quality metrics that might indicate manipulation.
        
        Args:
            image: Input image
            face_bbox: Face bounding box
            
        Returns:
            Dictionary of quality metrics
        """
        x, y, w, h = face_bbox
        face_region = image[y:y+h, x:x+w]
        
        quality_metrics = {}
        
        # Calculate sharpness
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        quality_metrics['sharpness'] = float(laplacian_var)
        
        # Calculate contrast
        quality_metrics['contrast'] = float(gray_face.std())
        
        # Calculate brightness
        quality_metrics['brightness'] = float(gray_face.mean())
        
        # Calculate color distribution
        color_std = np.std(face_region, axis=(0, 1))
        quality_metrics['color_variance'] = color_std.tolist()
        
        # Edge density
        edges = cv2.Canny(gray_face, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h)
        quality_metrics['edge_density'] = float(edge_density)
        
        return quality_metrics
