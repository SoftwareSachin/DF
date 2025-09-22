"""
Image processing utilities for deep fake detection.
Handles image loading, preprocessing, and basic analysis.
"""
import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional
from PIL import Image, ExifTags
import io
import base64

class ImageProcessor:
    """
    Handles image processing operations for deep fake detection.
    """
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    def load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Load image from bytes.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Image as numpy array in RGB format
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image from bytes")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image_rgb
    
    def load_image_from_file(self, image_path: str) -> np.ndarray:
        """
        Load image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array in RGB format
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image_rgb
    
    def extract_metadata(self, image_bytes: bytes) -> Dict:
        """
        Extract metadata from image.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary containing image metadata
        """
        metadata = {
            'format': None,
            'size': None,
            'mode': None,
            'exif': {},
            'creation_software': None,
            'camera_info': {},
            'gps_info': {}
        }
        
        try:
            # Use PIL to extract metadata
            image = Image.open(io.BytesIO(image_bytes))
            
            metadata['format'] = image.format
            metadata['size'] = image.size
            metadata['mode'] = image.mode
            
            # Extract EXIF data
            if hasattr(image, '_getexif') and image._getexif() is not None:
                exif_data = image._getexif()
                
                for tag_id, value in exif_data.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    
                    # Convert bytes to string if necessary
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except:
                            value = str(value)
                    
                    metadata['exif'][tag] = value
                    
                    # Extract specific information
                    if tag == 'Software':
                        metadata['creation_software'] = value
                    elif tag == 'Make':
                        metadata['camera_info']['make'] = value
                    elif tag == 'Model':
                        metadata['camera_info']['model'] = value
                    elif tag == 'DateTime':
                        metadata['camera_info']['datetime'] = value
        
        except Exception as e:
            print(f"Error extracting metadata: {e}")
        
        return metadata
    
    def analyze_image_properties(self, image: np.ndarray) -> Dict:
        """
        Analyze basic image properties.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary of image properties
        """
        properties = {
            'dimensions': image.shape,
            'channels': image.shape[2] if len(image.shape) == 3 else 1,
            'dtype': str(image.dtype),
            'size_mb': image.nbytes / (1024 * 1024),
            'color_stats': {},
            'quality_metrics': {}
        }
        
        # Color statistics
        if len(image.shape) == 3:
            for i, channel in enumerate(['R', 'G', 'B']):
                channel_data = image[:, :, i]
                properties['color_stats'][channel] = {
                    'mean': float(np.mean(channel_data)),
                    'std': float(np.std(channel_data)),
                    'min': int(np.min(channel_data)),
                    'max': int(np.max(channel_data))
                }
        else:
            # Grayscale image
            properties['color_stats']['gray'] = {
                'mean': float(np.mean(image)),
                'std': float(np.std(image)),
                'min': int(np.min(image)),
                'max': int(np.max(image))
            }
        
        # Quality metrics
        properties['quality_metrics'] = self._calculate_quality_metrics(image)
        
        return properties
    
    def _calculate_quality_metrics(self, image: np.ndarray) -> Dict:
        """Calculate image quality metrics."""
        # Convert to grayscale for some metrics
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        quality_metrics = {}
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        quality_metrics['sharpness'] = float(laplacian.var())
        
        # Contrast (standard deviation)
        quality_metrics['contrast'] = float(gray_image.std())
        
        # Brightness (mean intensity)
        quality_metrics['brightness'] = float(gray_image.mean())
        
        # Edge density
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges > 0) / (gray_image.shape[0] * gray_image.shape[1])
        quality_metrics['edge_density'] = float(edge_density)
        
        # Noise estimation (high frequency energy)
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_freq = cv2.filter2D(gray_image, -1, kernel)
        quality_metrics['noise_estimate'] = float(np.std(high_freq))
        
        # Color distribution (if color image)
        if len(image.shape) == 3:
            # Calculate color histogram entropy
            hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])
            
            # Normalize histograms
            hist_r = hist_r.flatten() / np.sum(hist_r)
            hist_g = hist_g.flatten() / np.sum(hist_g)
            hist_b = hist_b.flatten() / np.sum(hist_b)
            
            # Calculate entropy
            entropy_r = -np.sum(hist_r * np.log(hist_r + 1e-10))
            entropy_g = -np.sum(hist_g * np.log(hist_g + 1e-10))
            entropy_b = -np.sum(hist_b * np.log(hist_b + 1e-10))
            
            quality_metrics['color_entropy'] = {
                'red': float(entropy_r),
                'green': float(entropy_g),
                'blue': float(entropy_b),
                'average': float((entropy_r + entropy_g + entropy_b) / 3)
            }
        
        return quality_metrics
    
    def preprocess_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None, 
                        normalize: bool = True) -> np.ndarray:
        """
        Preprocess image for analysis.
        
        Args:
            image: Input image
            target_size: Optional target size (width, height)
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed image
        """
        processed_image = image.copy()
        
        # Resize if target size specified
        if target_size:
            processed_image = cv2.resize(processed_image, target_size)
        
        # Normalize pixel values
        if normalize:
            processed_image = processed_image.astype(np.float32) / 255.0
        
        return processed_image
    
    def detect_image_manipulations(self, image: np.ndarray) -> Dict:
        """
        Detect potential image manipulations.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of manipulation detection results
        """
        manipulation_analysis = {
            'copy_move': self._detect_copy_move(image),
            'splicing': self._detect_splicing(image),
            'resampling': self._detect_resampling(image),
            'noise_inconsistency': self._detect_noise_inconsistency(image),
            'illumination_inconsistency': self._detect_illumination_inconsistency(image)
        }
        
        return manipulation_analysis
    
    def _detect_copy_move(self, image: np.ndarray) -> Dict:
        """Detect copy-move forgery."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Use SIFT for keypoint detection
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        
        copy_move_result = {
            'suspicious_regions': 0,
            'keypoint_clusters': 0,
            'similarity_score': 0.0
        }
        
        if descriptors is not None and len(descriptors) > 10:
            # Use FLANN matcher to find similar keypoints
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(descriptors, descriptors, k=3)
            
            # Filter matches (exclude self-matches)
            good_matches = []
            for match_group in matches:
                if len(match_group) >= 2:
                    m, n = match_group[0], match_group[1]
                    if m.distance < 0.7 * n.distance and m.queryIdx != m.trainIdx:
                        good_matches.append(m)
            
            copy_move_result['similarity_score'] = len(good_matches) / len(descriptors) if len(descriptors) > 0 else 0.0
            
            # Cluster similar keypoints
            if len(good_matches) > 10:
                copy_move_result['suspicious_regions'] = 1
                copy_move_result['keypoint_clusters'] = len(good_matches)
        
        return copy_move_result
    
    def _detect_splicing(self, image: np.ndarray) -> Dict:
        """Detect image splicing."""
        splicing_result = {
            'edge_inconsistencies': 0,
            'color_inconsistencies': 0,
            'noise_inconsistencies': 0
        }
        
        # Convert to different color spaces for analysis
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Detect edges
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Analyze edge patterns
        # Look for abrupt changes in edge density
        kernel_size = 50
        height, width = gray_image.shape
        
        edge_densities = []
        for i in range(0, height - kernel_size, kernel_size // 2):
            for j in range(0, width - kernel_size, kernel_size // 2):
                region = edges[i:i+kernel_size, j:j+kernel_size]
                density = np.sum(region > 0) / (kernel_size * kernel_size)
                edge_densities.append(density)
        
        if edge_densities:
            edge_variance = np.var(edge_densities)
            if edge_variance > 0.01:  # Threshold for suspicious edge patterns
                splicing_result['edge_inconsistencies'] = 1
        
        # Analyze color consistency
        if len(image.shape) == 3:
            # Calculate local color statistics
            color_variances = []
            for i in range(0, image.shape[0] - kernel_size, kernel_size // 2):
                for j in range(0, image.shape[1] - kernel_size, kernel_size // 2):
                    region = image[i:i+kernel_size, j:j+kernel_size]
                    color_var = np.var(region, axis=(0, 1))
                    color_variances.append(np.mean(color_var))
            
            if color_variances:
                color_variance = np.var(color_variances)
                if color_variance > 100:  # Threshold for color inconsistency
                    splicing_result['color_inconsistencies'] = 1
        
        return splicing_result
    
    def _detect_resampling(self, image: np.ndarray) -> Dict:
        """Detect resampling artifacts."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        resampling_result = {
            'periodic_artifacts': 0,
            'interpolation_artifacts': 0,
            'aliasing_score': 0.0
        }
        
        # Apply FFT to detect periodic artifacts
        fft_result = np.fft.fft2(gray_image)
        fft_magnitude = np.abs(fft_result)
        
        # Look for periodic patterns in frequency domain
        fft_shifted = np.fft.fftshift(fft_magnitude)
        
        # Calculate radial average
        center = (fft_shifted.shape[0] // 2, fft_shifted.shape[1] // 2)
        y, x = np.ogrid[:fft_shifted.shape[0], :fft_shifted.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)
        
        # Find peaks in radial profile
        max_radius = min(center[0], center[1])
        radial_profile = []
        
        for radius in range(1, max_radius):
            mask = (r == radius)
            if np.any(mask):
                radial_profile.append(np.mean(fft_shifted[mask]))
        
        if len(radial_profile) > 10:
            # Look for periodic peaks
            profile_array = np.array(radial_profile)
            mean_profile = np.mean(profile_array)
            peaks = profile_array > (mean_profile + 2 * np.std(profile_array))
            
            if np.sum(peaks) > 3:
                resampling_result['periodic_artifacts'] = 1
        
        return resampling_result
    
    def _detect_noise_inconsistency(self, image: np.ndarray) -> Dict:
        """Detect noise inconsistencies."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Apply high-pass filter to extract noise
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        noise_image = cv2.filter2D(gray_image, -1, kernel)
        
        # Analyze noise in different regions
        kernel_size = 64
        height, width = gray_image.shape
        
        noise_variances = []
        for i in range(0, height - kernel_size, kernel_size // 2):
            for j in range(0, width - kernel_size, kernel_size // 2):
                region = noise_image[i:i+kernel_size, j:j+kernel_size]
                noise_var = np.var(region)
                noise_variances.append(noise_var)
        
        noise_consistency = {
            'variance_of_variances': float(np.var(noise_variances)) if noise_variances else 0.0,
            'inconsistent_regions': 0
        }
        
        if noise_variances:
            mean_noise_var = np.mean(noise_variances)
            std_noise_var = np.std(noise_variances)
            threshold = mean_noise_var + 2 * std_noise_var
            
            inconsistent_count = np.sum(np.array(noise_variances) > threshold)
            noise_consistency['inconsistent_regions'] = int(inconsistent_count)
        
        return noise_consistency
    
    def _detect_illumination_inconsistency(self, image: np.ndarray) -> Dict:
        """Detect illumination inconsistencies."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Analyze illumination patterns
        kernel_size = 64
        height, width = gray_image.shape
        
        brightness_values = []
        gradient_values = []
        
        for i in range(0, height - kernel_size, kernel_size // 2):
            for j in range(0, width - kernel_size, kernel_size // 2):
                brightness_region = gray_image[i:i+kernel_size, j:j+kernel_size]
                gradient_region = gradient_magnitude[i:i+kernel_size, j:j+kernel_size]
                
                brightness_values.append(np.mean(brightness_region))
                gradient_values.append(np.mean(gradient_region))
        
        illumination_analysis = {
            'brightness_variance': float(np.var(brightness_values)) if brightness_values else 0.0,
            'gradient_variance': float(np.var(gradient_values)) if gradient_values else 0.0,
            'inconsistent_illumination': 0
        }
        
        # Check for inconsistent illumination patterns
        if brightness_values and len(brightness_values) > 4:
            brightness_array = np.array(brightness_values)
            mean_brightness = np.mean(brightness_array)
            std_brightness = np.std(brightness_array)
            
            # Count regions with significantly different brightness
            outliers = np.abs(brightness_array - mean_brightness) > 2 * std_brightness
            illumination_analysis['inconsistent_illumination'] = int(np.sum(outliers))
        
        return illumination_analysis
    
    def convert_to_base64(self, image: np.ndarray, format: str = 'JPEG') -> str:
        """
        Convert image to base64 string.
        
        Args:
            image: Input image
            format: Output format (JPEG, PNG)
            
        Returns:
            Base64 encoded image string
        """
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Encode image
        if format.upper() == 'JPEG':
            _, buffer = cv2.imencode('.jpg', image_bgr)
        elif format.upper() == 'PNG':
            _, buffer = cv2.imencode('.png', image_bgr)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return image_base64
