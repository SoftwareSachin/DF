"""
Frequency domain analysis for deep fake detection.
Analyzes compression artifacts and frequency-based manipulation signatures.
"""
import numpy as np
import cv2
from typing import Dict, Tuple, List

# Optional imports with fallbacks
try:
    from scipy.fft import fft2, ifft2, fftfreq
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Using numpy fallbacks for frequency analysis.")
    # Use numpy fallbacks
    from numpy.fft import fft2, ifft2, fftfreq
    import numpy as ndimage  # This will fail, but we'll handle it
    ndimage = None

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting features disabled.")

try:
    from skimage import feature, measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Some feature detection will be limited.")

class FrequencyAnalyzer:
    """
    Analyzes frequency domain characteristics for deep fake detection.
    """
    
    def __init__(self, block_size: int = 8):
        self.block_size = block_size
        self.dct_coefficients = []
        
    def analyze_frequency_domain(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Comprehensive frequency domain analysis.
        
        Args:
            image: Input image
            face_bbox: Face bounding box (x, y, w, h)
            
        Returns:
            Dictionary of frequency analysis results
        """
        x, y, w, h = face_bbox
        face_region = image[y:y+h, x:x+w]
        
        analysis = {}
        
        # DCT block analysis
        analysis['dct_analysis'] = self._analyze_dct_blocks(face_region)
        
        # FFT analysis
        analysis['fft_analysis'] = self._analyze_fft_spectrum(face_region)
        
        # Compression artifact detection
        analysis['compression_artifacts'] = self._detect_compression_artifacts(face_region)
        
        # High frequency noise analysis
        analysis['noise_analysis'] = self._analyze_high_frequency_noise(face_region)
        
        # Spectral irregularities
        analysis['spectral_irregularities'] = self._detect_spectral_irregularities(face_region)
        
        # JPEG ghost detection
        analysis['jpeg_ghosts'] = self._detect_jpeg_ghosts(face_region)
        
        return analysis
    
    def _analyze_dct_blocks(self, image: np.ndarray) -> Dict:
        """Analyze DCT coefficients in 8x8 blocks."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        dct_analysis = {
            'dc_coefficients': [],
            'ac_energy': [],
            'block_artifacts': [],
            'quantization_artifacts': 0,
            'dct_histogram': None
        }
        
        height, width = gray_image.shape
        
        # Process 8x8 blocks
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                block = gray_image[i:i+self.block_size, j:j+self.block_size].astype(np.float32)
                
                # Apply DCT
                dct_block = cv2.dct(block)
                
                # Extract DC coefficient
                dc_coeff = dct_block[0, 0]
                dct_analysis['dc_coefficients'].append(float(dc_coeff))
                
                # Calculate AC energy
                ac_energy = np.sum(dct_block[1:, :]**2) + np.sum(dct_block[0, 1:]**2)
                dct_analysis['ac_energy'].append(float(ac_energy))
                
                # Detect quantization artifacts
                if self._detect_quantization_artifacts(dct_block):
                    dct_analysis['quantization_artifacts'] += 1
        
        # Calculate DCT coefficient statistics
        if dct_analysis['dc_coefficients']:
            dc_array = np.array(dct_analysis['dc_coefficients'])
            dct_analysis['dc_mean'] = float(np.mean(dc_array))
            dct_analysis['dc_variance'] = float(np.var(dc_array))
            dct_analysis['dc_range'] = float(np.max(dc_array) - np.min(dc_array))
            
            # Create histogram of DC coefficients
            hist, bins = np.histogram(dc_array, bins=50)
            dct_analysis['dct_histogram'] = {
                'counts': hist.tolist(),
                'bins': bins.tolist()
            }
        
        return dct_analysis
    
    def _detect_quantization_artifacts(self, dct_block: np.ndarray) -> bool:
        """Detect quantization artifacts in DCT block."""
        # Check for zero coefficients in high frequency components
        high_freq_zeros = np.sum(dct_block[4:, 4:] == 0)
        total_high_freq = (self.block_size - 4) ** 2
        
        # If more than 70% of high frequency coefficients are zero, likely quantized
        return (high_freq_zeros / total_high_freq) > 0.7
    
    def _analyze_fft_spectrum(self, image: np.ndarray) -> Dict:
        """Analyze FFT spectrum characteristics."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Compute 2D FFT
        fft_result = fft2(gray_image)
        fft_magnitude = np.abs(fft_result)
        fft_phase = np.angle(fft_result)
        
        # Shift zero frequency to center
        fft_shifted = np.fft.fftshift(fft_magnitude)
        
        fft_analysis = {
            'spectral_energy': float(np.sum(fft_magnitude**2)),
            'spectral_centroid': self._calculate_spectral_centroid(fft_shifted),
            'spectral_rolloff': self._calculate_spectral_rolloff(fft_shifted),
            'spectral_flatness': self._calculate_spectral_flatness(fft_shifted),
            'phase_coherence': self._analyze_phase_coherence(fft_phase),
            'frequency_peaks': self._detect_frequency_peaks(fft_shifted)
        }
        
        return fft_analysis
    
    def _calculate_spectral_centroid(self, spectrum: np.ndarray) -> float:
        """Calculate spectral centroid."""
        height, width = spectrum.shape
        
        # Create frequency grids
        freq_y = np.arange(height) - height // 2
        freq_x = np.arange(width) - width // 2
        
        # Calculate weighted centroid
        total_energy = np.sum(spectrum)
        if total_energy == 0:
            return 0.0
        
        centroid_y = np.sum(np.sum(spectrum, axis=1) * freq_y) / total_energy
        centroid_x = np.sum(np.sum(spectrum, axis=0) * freq_x) / total_energy
        
        # Return magnitude of centroid
        return float(np.sqrt(centroid_x**2 + centroid_y**2))
    
    def _calculate_spectral_rolloff(self, spectrum: np.ndarray, threshold: float = 0.85) -> float:
        """Calculate spectral rolloff point."""
        # Flatten spectrum and sort
        flat_spectrum = spectrum.flatten()
        sorted_indices = np.argsort(flat_spectrum)[::-1]
        
        total_energy = np.sum(flat_spectrum)
        cumulative_energy = 0
        
        for i, idx in enumerate(sorted_indices):
            cumulative_energy += flat_spectrum[idx]
            if cumulative_energy >= threshold * total_energy:
                return float(i / len(flat_spectrum))
        
        return 1.0
    
    def _calculate_spectral_flatness(self, spectrum: np.ndarray) -> float:
        """Calculate spectral flatness (Wiener entropy)."""
        # Avoid log of zero
        spectrum_safe = spectrum + 1e-10
        
        # Calculate geometric and arithmetic means
        geometric_mean = np.exp(np.mean(np.log(spectrum_safe)))
        arithmetic_mean = np.mean(spectrum_safe)
        
        if arithmetic_mean == 0:
            return 0.0
        
        return float(geometric_mean / arithmetic_mean)
    
    def _analyze_phase_coherence(self, phase: np.ndarray) -> Dict:
        """Analyze phase coherence patterns."""
        # Calculate phase derivatives
        phase_grad_x = np.diff(phase, axis=1)
        phase_grad_y = np.diff(phase, axis=0)
        
        # Wrap phase differences to [-π, π]
        phase_grad_x = np.arctan2(np.sin(phase_grad_x), np.cos(phase_grad_x))
        phase_grad_y = np.arctan2(np.sin(phase_grad_y), np.cos(phase_grad_y))
        
        coherence = {
            'phase_variance_x': float(np.var(phase_grad_x)),
            'phase_variance_y': float(np.var(phase_grad_y)),
            'phase_smoothness': float(1.0 / (1.0 + np.var(phase_grad_x) + np.var(phase_grad_y)))
        }
        
        return coherence
    
    def _detect_frequency_peaks(self, spectrum: np.ndarray) -> Dict:
        """Detect suspicious frequency peaks."""
        # Find local maxima
        local_maxima = feature.peak_local_maxima(spectrum, min_distance=5, threshold_abs=np.mean(spectrum))
        
        peaks_info = {
            'num_peaks': len(local_maxima[0]) if local_maxima[0].size > 0 else 0,
            'peak_strengths': [],
            'peak_locations': []
        }
        
        if local_maxima[0].size > 0:
            for i, j in zip(local_maxima[0], local_maxima[1]):
                peak_strength = spectrum[i, j]
                peaks_info['peak_strengths'].append(float(peak_strength))
                peaks_info['peak_locations'].append([int(i), int(j)])
        
        return peaks_info
    
    def _detect_compression_artifacts(self, image: np.ndarray) -> Dict:
        """Detect compression-related artifacts."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        artifacts = {
            'blocking_artifacts': self._detect_blocking_artifacts(gray_image),
            'ringing_artifacts': self._detect_ringing_artifacts(gray_image),
            'mosquito_noise': self._detect_mosquito_noise(gray_image),
            'compression_ratio_estimate': self._estimate_compression_ratio(gray_image)
        }
        
        return artifacts
    
    def _detect_blocking_artifacts(self, image: np.ndarray) -> Dict:
        """Detect 8x8 blocking artifacts."""
        height, width = image.shape
        
        # Calculate differences across block boundaries
        vertical_diffs = []
        horizontal_diffs = []
        
        # Vertical block boundaries
        for i in range(self.block_size, height, self.block_size):
            if i < height:
                diff = np.mean(np.abs(image[i-1, :] - image[i, :]))
                vertical_diffs.append(diff)
        
        # Horizontal block boundaries
        for j in range(self.block_size, width, self.block_size):
            if j < width:
                diff = np.mean(np.abs(image[:, j-1] - image[:, j]))
                horizontal_diffs.append(diff)
        
        blocking_score = 0.0
        if vertical_diffs and horizontal_diffs:
            # Compare boundary differences to internal differences
            all_vertical_diffs = []
            all_horizontal_diffs = []
            
            for i in range(1, height):
                diff = np.mean(np.abs(image[i-1, :] - image[i, :]))
                all_vertical_diffs.append(diff)
            
            for j in range(1, width):
                diff = np.mean(np.abs(image[:, j-1] - image[:, j]))
                all_horizontal_diffs.append(diff)
            
            if all_vertical_diffs and all_horizontal_diffs:
                boundary_mean = np.mean(vertical_diffs + horizontal_diffs)
                overall_mean = np.mean(all_vertical_diffs + all_horizontal_diffs)
                
                if overall_mean > 0:
                    blocking_score = boundary_mean / overall_mean
        
        return {
            'blocking_score': float(blocking_score),
            'vertical_artifacts': len(vertical_diffs),
            'horizontal_artifacts': len(horizontal_diffs)
        }
    
    def _detect_ringing_artifacts(self, image: np.ndarray) -> float:
        """Detect ringing artifacts around edges."""
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Dilate edges to create regions around edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_regions = cv2.dilate(edges, kernel, iterations=1)
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate variance in edge regions
        edge_variance = np.var(gradient_magnitude[edge_regions > 0])
        
        return float(edge_variance)
    
    def _detect_mosquito_noise(self, image: np.ndarray) -> float:
        """Detect mosquito noise patterns."""
        # Apply high-pass filter
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_freq = cv2.filter2D(image, -1, kernel)
        
        # Calculate noise metric
        noise_energy = np.mean(np.abs(high_freq))
        
        return float(noise_energy)
    
    def _estimate_compression_ratio(self, image: np.ndarray) -> float:
        """Estimate JPEG compression ratio."""
        # Calculate image complexity
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        
        # Calculate texture complexity
        glcm = self._calculate_glcm(image)
        texture_complexity = np.sum(glcm * np.arange(256)[:, np.newaxis])
        
        # Estimate compression ratio based on complexity
        complexity_score = edge_density + texture_complexity / 1000.0
        estimated_ratio = min(95.0, max(10.0, 100.0 - complexity_score * 50.0))
        
        return float(estimated_ratio)
    
    def _calculate_glcm(self, image: np.ndarray) -> np.ndarray:
        """Calculate Gray Level Co-occurrence Matrix."""
        # Simple GLCM implementation
        max_val = 256
        glcm = np.zeros((max_val, max_val))
        
        rows, cols = image.shape
        for i in range(rows - 1):
            for j in range(cols - 1):
                pixel1 = image[i, j]
                pixel2 = image[i, j + 1]  # Horizontal neighbor
                glcm[pixel1, pixel2] += 1
        
        # Normalize
        glcm = glcm / np.sum(glcm)
        
        return glcm
    
    def _analyze_high_frequency_noise(self, image: np.ndarray) -> Dict:
        """Analyze high frequency noise characteristics."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply different high-pass filters
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        noise_analysis = {
            'laplacian_variance': float(laplacian.var()),
            'sobel_energy': float(np.sum(sobel_x**2 + sobel_y**2)),
            'noise_distribution': self._analyze_noise_distribution(laplacian),
            'snr_estimate': self._estimate_snr(gray_image, laplacian)
        }
        
        return noise_analysis
    
    def _analyze_noise_distribution(self, noise_image: np.ndarray) -> Dict:
        """Analyze noise distribution characteristics."""
        # Calculate histogram of noise
        hist, bins = np.histogram(noise_image.flatten(), bins=50, density=True)
        
        # Calculate distribution metrics
        mean_noise = np.mean(noise_image)
        std_noise = np.std(noise_image)
        skewness = np.mean(((noise_image - mean_noise) / std_noise)**3) if std_noise > 0 else 0
        
        return {
            'noise_mean': float(mean_noise),
            'noise_std': float(std_noise),
            'noise_skewness': float(skewness),
            'histogram_entropy': float(-np.sum(hist * np.log(hist + 1e-10)))
        }
    
    def _estimate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio."""
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)
    
    def _detect_spectral_irregularities(self, image: np.ndarray) -> Dict:
        """Detect spectral irregularities that might indicate manipulation."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Compute FFT
        fft_result = fft2(gray_image)
        magnitude_spectrum = np.abs(fft_result)
        
        irregularities = {
            'spectral_anomalies': self._detect_spectral_anomalies(magnitude_spectrum),
            'frequency_gaps': self._detect_frequency_gaps(magnitude_spectrum),
            'artificial_peaks': self._detect_artificial_peaks(magnitude_spectrum)
        }
        
        return irregularities
    
    def _detect_spectral_anomalies(self, spectrum: np.ndarray) -> int:
        """Detect spectral anomalies."""
        # Calculate radial average
        center = (spectrum.shape[0] // 2, spectrum.shape[1] // 2)
        radial_profile = self._calculate_radial_profile(spectrum, center)
        
        # Detect anomalies in radial profile
        anomaly_count = 0
        if len(radial_profile) > 10:
            # Use median filter to detect outliers
            filtered_profile = ndimage.median_filter(radial_profile, size=3)
            differences = np.abs(radial_profile - filtered_profile)
            threshold = np.std(differences) * 3
            anomaly_count = np.sum(differences > threshold)
        
        return int(anomaly_count)
    
    def _calculate_radial_profile(self, spectrum: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """Calculate radial profile of spectrum."""
        y, x = np.ogrid[:spectrum.shape[0], :spectrum.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)
        
        # Calculate mean values at each radius
        max_radius = min(center[0], center[1], spectrum.shape[0] - center[0], spectrum.shape[1] - center[1])
        radial_profile = []
        
        for radius in range(max_radius):
            mask = (r == radius)
            if np.any(mask):
                radial_profile.append(np.mean(spectrum[mask]))
        
        return np.array(radial_profile)
    
    def _detect_frequency_gaps(self, spectrum: np.ndarray) -> int:
        """Detect suspicious frequency gaps."""
        # Look for regions with unusually low energy
        threshold = np.percentile(spectrum, 10)  # Bottom 10%
        low_energy_regions = spectrum < threshold
        
        # Count connected components of low energy regions
        labeled_regions, num_regions = ndimage.label(low_energy_regions)
        
        # Filter out small regions
        large_gaps = 0
        for region_id in range(1, num_regions + 1):
            region_size = np.sum(labeled_regions == region_id)
            if region_size > 100:  # Threshold for significant gaps
                large_gaps += 1
        
        return large_gaps
    
    def _detect_artificial_peaks(self, spectrum: np.ndarray) -> int:
        """Detect artificial frequency peaks."""
        # Find peaks that are significantly higher than their neighborhood
        mean_spectrum = np.mean(spectrum)
        std_spectrum = np.std(spectrum)
        threshold = mean_spectrum + 5 * std_spectrum
        
        # Find peaks above threshold
        peaks = spectrum > threshold
        
        # Count isolated peaks (potential artifacts)
        kernel = np.ones((3, 3))
        dilated_peaks = ndimage.binary_dilation(peaks, kernel)
        isolated_peaks = peaks & ~ndimage.binary_erosion(dilated_peaks, kernel)
        
        return int(np.sum(isolated_peaks))
    
    def _detect_jpeg_ghosts(self, image: np.ndarray) -> Dict:
        """Detect JPEG ghosts indicating multiple compression."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        ghost_analysis = {
            'ghost_score': 0.0,
            'compression_levels': [],
            'quality_estimates': []
        }
        
        # Test multiple JPEG quality levels
        quality_levels = [70, 80, 90, 95]
        ghost_scores = []
        
        for quality in quality_levels:
            # Simulate JPEG compression
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded_img = cv2.imencode('.jpg', gray_image, encode_param)
            decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)
            
            # Calculate difference
            if decoded_img is not None and decoded_img.shape == gray_image.shape:
                diff = np.abs(gray_image.astype(float) - decoded_img.astype(float))
                ghost_score = np.mean(diff)
                ghost_scores.append(ghost_score)
            else:
                ghost_scores.append(float('inf'))
        
        # Find minimum ghost score (best matching quality)
        if ghost_scores:
            min_ghost_idx = np.argmin(ghost_scores)
            ghost_analysis['ghost_score'] = float(ghost_scores[min_ghost_idx])
            ghost_analysis['estimated_quality'] = quality_levels[min_ghost_idx]
        
        return ghost_analysis
