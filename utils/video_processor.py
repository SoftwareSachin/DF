"""
Video processing utilities for deep fake detection.
Handles video loading, frame extraction, and preprocessing.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Generator
import tempfile
import os
from pathlib import Path

class VideoProcessor:
    """
    Handles video processing operations for deep fake detection.
    """
    
    def __init__(self, max_frames: int = 30, frame_skip: int = 5):
        self.max_frames = max_frames
        self.frame_skip = frame_skip
        self.temp_dir = tempfile.mkdtemp()
    
    def extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], dict]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (frames_list, video_metadata)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video metadata
        metadata = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': 0,
            'codec': self._get_codec_info(cap)
        }
        
        if metadata['fps'] > 0:
            metadata['duration'] = metadata['frame_count'] / metadata['fps']
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while extracted_count < self.max_frames and cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Skip frames based on frame_skip parameter
            if frame_count % self.frame_skip == 0:
                frames.append(frame.copy())
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        
        if not frames:
            raise ValueError("No frames could be extracted from the video")
        
        return frames, metadata
    
    def extract_frames_from_bytes(self, video_bytes: bytes) -> Tuple[List[np.ndarray], dict]:
        """
        Extract frames from video bytes.
        
        Args:
            video_bytes: Video file as bytes
            
        Returns:
            Tuple of (frames_list, video_metadata)
        """
        # Save bytes to temporary file
        temp_video_path = os.path.join(self.temp_dir, 'temp_video.mp4')
        
        with open(temp_video_path, 'wb') as f:
            f.write(video_bytes)
        
        try:
            frames, metadata = self.extract_frames(temp_video_path)
            return frames, metadata
        finally:
            # Clean up temporary file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
    
    def _get_codec_info(self, cap: cv2.VideoCapture) -> str:
        """Get codec information from video capture."""
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
        return codec.strip()
    
    def preprocess_frames(self, frames: List[np.ndarray], target_size: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        Preprocess video frames.
        
        Args:
            frames: List of video frames
            target_size: Optional target size for resizing (width, height)
            
        Returns:
            List of preprocessed frames
        """
        processed_frames = []
        
        for frame in frames:
            processed_frame = frame.copy()
            
            # Resize if target size specified
            if target_size:
                processed_frame = cv2.resize(processed_frame, target_size)
            
            # Ensure frame is in correct format
            if len(processed_frame.shape) == 3 and processed_frame.shape[2] == 3:
                # Convert BGR to RGB for processing
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            processed_frames.append(processed_frame)
        
        return processed_frames
    
    def analyze_video_quality(self, frames: List[np.ndarray]) -> dict:
        """
        Analyze video quality metrics.
        
        Args:
            frames: List of video frames
            
        Returns:
            Dictionary of quality metrics
        """
        quality_metrics = {
            'frame_count': len(frames),
            'resolution_consistency': True,
            'brightness_stats': {},
            'contrast_stats': {},
            'sharpness_stats': {},
            'color_stats': {}
        }
        
        if not frames:
            return quality_metrics
        
        # Check resolution consistency
        base_shape = frames[0].shape
        for frame in frames[1:]:
            if frame.shape != base_shape:
                quality_metrics['resolution_consistency'] = False
                break
        
        # Calculate quality metrics for each frame
        brightness_values = []
        contrast_values = []
        sharpness_values = []
        color_variances = []
        
        for frame in frames:
            # Convert to grayscale for some metrics
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
            
            # Brightness (mean intensity)
            brightness = np.mean(gray_frame)
            brightness_values.append(brightness)
            
            # Contrast (standard deviation)
            contrast = np.std(gray_frame)
            contrast_values.append(contrast)
            
            # Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_values.append(sharpness)
            
            # Color variance (if color image)
            if len(frame.shape) == 3:
                color_var = np.var(frame, axis=(0, 1))
                color_variances.append(color_var)
        
        # Calculate statistics
        quality_metrics['brightness_stats'] = {
            'mean': float(np.mean(brightness_values)),
            'std': float(np.std(brightness_values)),
            'min': float(np.min(brightness_values)),
            'max': float(np.max(brightness_values))
        }
        
        quality_metrics['contrast_stats'] = {
            'mean': float(np.mean(contrast_values)),
            'std': float(np.std(contrast_values)),
            'min': float(np.min(contrast_values)),
            'max': float(np.max(contrast_values))
        }
        
        quality_metrics['sharpness_stats'] = {
            'mean': float(np.mean(sharpness_values)),
            'std': float(np.std(sharpness_values)),
            'min': float(np.min(sharpness_values)),
            'max': float(np.max(sharpness_values))
        }
        
        if color_variances:
            color_array = np.array(color_variances)
            quality_metrics['color_stats'] = {
                'mean_variance': color_array.mean(axis=0).tolist(),
                'std_variance': color_array.std(axis=0).tolist()
            }
        
        return quality_metrics
    
    def detect_scene_changes(self, frames: List[np.ndarray], threshold: float = 0.3) -> List[int]:
        """
        Detect scene changes in video frames.
        
        Args:
            frames: List of video frames
            threshold: Threshold for scene change detection
            
        Returns:
            List of frame indices where scene changes occur
        """
        if len(frames) < 2:
            return []
        
        scene_changes = [0]  # First frame is always a scene change
        
        for i in range(1, len(frames)):
            # Convert frames to grayscale
            gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY) if len(frames[i-1].shape) == 3 else frames[i-1]
            gray2 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) if len(frames[i].shape) == 3 else frames[i]
            
            # Resize to same size if different
            if gray1.shape != gray2.shape:
                min_h, min_w = min(gray1.shape[0], gray2.shape[0]), min(gray1.shape[1], gray2.shape[1])
                gray1 = cv2.resize(gray1, (min_w, min_h))
                gray2 = cv2.resize(gray2, (min_w, min_h))
            
            # Calculate frame difference
            diff = np.abs(gray1.astype(float) - gray2.astype(float))
            mean_diff = np.mean(diff) / 255.0  # Normalize to [0, 1]
            
            if mean_diff > threshold:
                scene_changes.append(i)
        
        return scene_changes
    
    def extract_audio_features(self, video_path: str) -> dict:
        """
        Extract basic audio features from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary of audio features
        """
        cap = cv2.VideoCapture(video_path)
        
        audio_features = {
            'has_audio': False,
            'audio_codec': None,
            'sample_rate': None,
            'channels': None
        }
        
        # Check if video has audio track
        # Note: OpenCV has limited audio support, this is a basic check
        try:
            # This is a simplified check - in production you'd use libraries like librosa or moviepy
            audio_features['has_audio'] = True  # Assume audio present for now
        except:
            pass
        
        cap.release()
        return audio_features
    
    def calculate_motion_vectors(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Calculate motion vectors between consecutive frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            List of motion vector arrays
        """
        if len(frames) < 2:
            return []
        
        motion_vectors = []
        
        # Parameters for optical flow
        lk_params = dict(winSize=(15, 15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        for i in range(len(frames) - 1):
            # Convert to grayscale
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) if len(frames[i].shape) == 3 else frames[i]
            gray2 = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY) if len(frames[i+1].shape) == 3 else frames[i+1]
            
            # Detect feature points in first frame
            p0 = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            
            if p0 is not None:
                # Calculate optical flow
                p1, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
                
                # Filter good points
                good_new = p1[status == 1]
                good_old = p0[status == 1]
                
                # Calculate motion vectors
                motion = good_new - good_old
                motion_vectors.append(motion)
            else:
                motion_vectors.append(np.array([]))
        
        return motion_vectors
    
    def analyze_compression_artifacts(self, frames: List[np.ndarray]) -> dict:
        """
        Analyze compression artifacts in video frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            Dictionary of compression analysis
        """
        compression_analysis = {
            'avg_jpeg_quality': 0.0,
            'compression_consistency': 0.0,
            'artifact_detection': []
        }
        
        quality_estimates = []
        
        for i, frame in enumerate(frames):
            # Convert to grayscale for analysis
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
            
            # Estimate JPEG quality
            quality_estimate = self._estimate_jpeg_quality(gray_frame)
            quality_estimates.append(quality_estimate)
            
            # Detect blocking artifacts
            blocking_score = self._detect_blocking_artifacts(gray_frame)
            
            compression_analysis['artifact_detection'].append({
                'frame_index': i,
                'quality_estimate': quality_estimate,
                'blocking_score': blocking_score
            })
        
        # Calculate average quality and consistency
        if quality_estimates:
            compression_analysis['avg_jpeg_quality'] = float(np.mean(quality_estimates))
            compression_analysis['compression_consistency'] = float(1.0 / (1.0 + np.var(quality_estimates)))
        
        return compression_analysis
    
    def _estimate_jpeg_quality(self, image: np.ndarray) -> float:
        """Estimate JPEG quality of an image."""
        # Calculate image complexity metrics
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        complexity = laplacian.var()
        
        # Simple quality estimation based on complexity
        # Higher complexity usually indicates lower compression
        quality_estimate = min(95.0, max(10.0, complexity / 100.0))
        
        return quality_estimate
    
    def _detect_blocking_artifacts(self, image: np.ndarray, block_size: int = 8) -> float:
        """Detect blocking artifacts in image."""
        height, width = image.shape
        
        # Calculate differences at block boundaries
        boundary_diffs = []
        
        # Vertical boundaries
        for i in range(block_size, height, block_size):
            if i < height:
                diff = np.mean(np.abs(image[i-1, :].astype(float) - image[i, :].astype(float)))
                boundary_diffs.append(diff)
        
        # Horizontal boundaries
        for j in range(block_size, width, block_size):
            if j < width:
                diff = np.mean(np.abs(image[:, j-1].astype(float) - image[:, j].astype(float)))
                boundary_diffs.append(diff)
        
        return float(np.mean(boundary_diffs)) if boundary_diffs else 0.0
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
    
    def __del__(self):
        """Destructor to clean up resources."""
        self.cleanup()
