"""
Enhanced Deep Fake Detection with Advanced Real vs Fake Analysis
"""
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import time
import logging
from scipy import stats
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class AuthenticityAnalyzer:
    """Advanced analyzer to distinguish real photos from AI-generated content"""
    
    def __init__(self):
        self.noise_patterns = {}
        
    def analyze_image_authenticity(self, image: np.ndarray) -> Dict:
        """
        Comprehensive analysis to determine if image is real or AI-generated
        """
        start_time = time.time()
        
        try:
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Multiple analysis layers
            noise_analysis = self._analyze_noise_patterns(gray)
            color_analysis = self._analyze_color_distribution(image, hsv, lab)
            texture_analysis = self._analyze_texture_realism(gray)
            edge_analysis = self._analyze_edge_consistency(gray)
            frequency_analysis = self._analyze_frequency_domain(gray)
            statistical_analysis = self._analyze_statistical_properties(image)
            
            # Combine all analyses
            authenticity_scores = [
                noise_analysis['authenticity'],
                color_analysis['authenticity'],
                texture_analysis['authenticity'],
                edge_analysis['authenticity'],
                frequency_analysis['authenticity'],
                statistical_analysis['authenticity']
            ]
            
            overall_authenticity = np.mean(authenticity_scores)
            
            # Determine likely source
            image_type = self._determine_image_type(
                noise_analysis, color_analysis, texture_analysis
            )
            
            inference_time = time.time() - start_time
            
            return {
                'model': 'Enhanced Authenticity Analyzer',
                'authenticity_score': overall_authenticity,
                'is_real': overall_authenticity > 0.5,
                'confidence': abs(overall_authenticity - 0.5) * 2,  # Convert to 0-1 range
                'inference_time': inference_time,
                'detailed_analysis': {
                    'noise_analysis': noise_analysis,
                    'color_analysis': color_analysis,
                    'texture_analysis': texture_analysis,
                    'edge_analysis': edge_analysis,
                    'frequency_analysis': frequency_analysis,
                    'statistical_analysis': statistical_analysis
                },
                'image_type': image_type,
                'quality_indicators': self._get_quality_indicators(
                    noise_analysis, color_analysis, texture_analysis
                ),
                'recommendation': self._get_recommendation(overall_authenticity, image_type)
            }
            
        except Exception as e:
            logger.error(f"Authenticity analysis failed: {e}")
            return {
                'model': 'Enhanced Authenticity Analyzer',
                'authenticity_score': 0.5,
                'is_real': False,
                'confidence': 0.0,
                'inference_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _analyze_noise_patterns(self, gray: np.ndarray) -> Dict:
        """Analyze noise patterns - real photos have characteristic sensor noise"""
        # Calculate local variance to detect noise
        kernel = np.ones((3, 3), np.float32) / 9
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        # Real photos have more varied noise patterns
        noise_variance = np.var(local_variance)
        noise_mean = np.mean(local_variance)
        
        # High-frequency noise analysis
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        high_freq_noise = np.var(laplacian)
        
        # Real photos typically have more natural noise
        authenticity = min(1.0, (noise_variance + high_freq_noise) / 1000)
        
        return {
            'authenticity': authenticity,
            'noise_variance': float(noise_variance),
            'noise_mean': float(noise_mean),
            'high_freq_noise': float(high_freq_noise),
            'assessment': 'Natural noise patterns' if authenticity > 0.6 else 'Artificial/smooth patterns'
        }
    
    def _analyze_color_distribution(self, image: np.ndarray, hsv: np.ndarray, lab: np.ndarray) -> Dict:
        """Analyze color distribution patterns"""
        # Color histogram analysis
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
        
        # Calculate color distribution entropy
        color_entropy = 0
        for hist in [hist_b, hist_g, hist_r]:
            hist_norm = hist / (hist.sum() + 1e-7)
            color_entropy += stats.entropy(hist_norm.flatten() + 1e-7)
        
        # Saturation analysis in HSV
        saturation_mean = np.mean(hsv[:, :, 1])
        saturation_var = np.var(hsv[:, :, 1])
        
        # Real photos have more natural color distributions
        authenticity = min(1.0, (color_entropy + saturation_var / 100) / 20)
        
        return {
            'authenticity': authenticity,
            'color_entropy': float(color_entropy),
            'saturation_mean': float(saturation_mean),
            'saturation_variance': float(saturation_var),
            'assessment': 'Natural color distribution' if authenticity > 0.5 else 'Artificial color patterns'
        }
    
    def _analyze_texture_realism(self, gray: np.ndarray) -> Dict:
        """Analyze texture patterns for realism"""
        # Local Binary Pattern analysis
        def local_binary_pattern(image, radius=1, n_points=8):
            rows, cols = image.shape
            lbp = np.zeros((rows, cols), dtype=np.uint8)
            
            # Simple LBP implementation
            for i in range(radius, rows - radius):
                for j in range(radius, cols - radius):
                    center = image[i, j]
                    binary_pattern = 0
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(radius * np.cos(angle))
                        y = int(radius * np.sin(angle))
                        neighbor = image[i + y, j + x]
                        if neighbor >= center:
                            binary_pattern |= (1 << k)
                    lbp[i, j] = binary_pattern
            return lbp
        
        lbp = local_binary_pattern(gray)
        lbp_variance = np.var(lbp)
        
        # Edge density analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Texture complexity
        texture_complexity = lbp_variance * edge_density
        
        # Real textures are typically more complex
        authenticity = min(1.0, texture_complexity / 500)
        
        return {
            'authenticity': authenticity,
            'lbp_variance': float(lbp_variance),
            'edge_density': float(edge_density),
            'texture_complexity': float(texture_complexity),
            'assessment': 'Natural texture patterns' if authenticity > 0.5 else 'Simplified/artificial textures'
        }
    
    def _analyze_edge_consistency(self, gray: np.ndarray) -> Dict:
        """Analyze edge consistency and quality"""
        # Multiple edge detection methods
        canny_edges = cv2.Canny(gray, 50, 150)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Edge strength consistency
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_consistency = np.std(edge_magnitude[edge_magnitude > 0])
        
        # Real photos have more varied edge strengths
        authenticity = min(1.0, edge_consistency / 100)
        
        return {
            'authenticity': authenticity,
            'edge_consistency': float(edge_consistency),
            'canny_edge_count': int(np.sum(canny_edges > 0)),
            'assessment': 'Natural edge patterns' if authenticity > 0.5 else 'Artificial edge consistency'
        }
    
    def _analyze_frequency_domain(self, gray: np.ndarray) -> Dict:
        """Analyze frequency domain characteristics"""
        # FFT analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Analyze frequency distribution
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Low, mid, and high frequency energy
        low_freq = magnitude_spectrum[center_y-h//4:center_y+h//4, center_x-w//4:center_x+w//4]
        high_freq = magnitude_spectrum[:center_y//2, :center_x//2]
        
        low_energy = np.mean(low_freq)
        high_energy = np.mean(high_freq)
        
        # Real photos have more balanced frequency distribution
        freq_ratio = high_energy / (low_energy + 1e-7)
        authenticity = min(1.0, freq_ratio)
        
        return {
            'authenticity': authenticity,
            'low_freq_energy': float(low_energy),
            'high_freq_energy': float(high_energy),
            'frequency_ratio': float(freq_ratio),
            'assessment': 'Natural frequency distribution' if authenticity > 0.3 else 'Compressed/artificial frequencies'
        }
    
    def _analyze_statistical_properties(self, image: np.ndarray) -> Dict:
        """Analyze statistical properties of the image"""
        # Convert to float for analysis
        img_float = image.astype(np.float32) / 255.0
        
        # Color channel correlations
        correlations = []
        for i in range(3):
            for j in range(i+1, 3):
                corr = np.corrcoef(img_float[:,:,i].flatten(), img_float[:,:,j].flatten())[0,1]
                correlations.append(abs(corr))
        
        avg_correlation = np.mean(correlations)
        
        # Pixel value distribution
        pixel_entropy = stats.entropy(cv2.calcHist([image], [0,1,2], None, [16,16,16], [0,256,0,256,0,256]).flatten() + 1e-7)
        
        # Real photos have lower correlation and higher entropy
        authenticity = min(1.0, pixel_entropy / 10 + (1 - avg_correlation))
        
        return {
            'authenticity': authenticity,
            'avg_correlation': float(avg_correlation),
            'pixel_entropy': float(pixel_entropy),
            'assessment': 'Natural statistical properties' if authenticity > 0.5 else 'Artificial statistical patterns'
        }
    
    def _determine_image_type(self, noise_analysis: Dict, color_analysis: Dict, texture_analysis: Dict) -> str:
        """Determine the likely source/type of the image"""
        noise_score = noise_analysis['authenticity']
        color_score = color_analysis['authenticity']
        texture_score = texture_analysis['authenticity']
        
        if noise_score > 0.7 and color_score > 0.6 and texture_score > 0.6:
            return "professional_camera"
        elif noise_score > 0.5 and color_score > 0.5:
            return "smartphone_photo"
        elif noise_score < 0.3 and texture_score < 0.4:
            return "ai_generated"
        elif color_score < 0.4:
            return "heavily_edited"
        else:
            return "unknown_origin"
    
    def _get_quality_indicators(self, noise_analysis: Dict, color_analysis: Dict, texture_analysis: Dict) -> List[str]:
        """Get list of quality indicators"""
        indicators = []
        
        if noise_analysis['authenticity'] > 0.6:
            indicators.append("Natural sensor noise detected")
        if color_analysis['authenticity'] > 0.6:
            indicators.append("Realistic color distribution")
        if texture_analysis['authenticity'] > 0.6:
            indicators.append("Complex natural textures")
        
        if noise_analysis['authenticity'] < 0.3:
            indicators.append("Artificial smoothing detected")
        if color_analysis['authenticity'] < 0.3:
            indicators.append("Unnatural color patterns")
        if texture_analysis['authenticity'] < 0.3:
            indicators.append("Simplified textures")
        
        return indicators
    
    def _get_recommendation(self, authenticity_score: float, image_type: str) -> str:
        """Get recommendation based on analysis"""
        if authenticity_score > 0.8:
            return f"High confidence this is a real {image_type.replace('_', ' ')}"
        elif authenticity_score > 0.6:
            return f"Likely a real {image_type.replace('_', ' ')} with minor concerns"
        elif authenticity_score > 0.4:
            return f"Uncertain - could be edited {image_type.replace('_', ' ')} or AI-generated"
        elif authenticity_score > 0.2:
            return "Likely AI-generated or heavily manipulated content"
        else:
            return "High probability of AI generation or deep fake content"