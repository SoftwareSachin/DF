"""
Simplified detection implementation
"""
import numpy as np
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SimpleDetector:
    """Base class for simplified detectors"""
    
    def __init__(self, name: str):
        self.name = name
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Simple detection that returns mock results for UI testing"""
        # Simple heuristic: check image properties
        height, width = image.shape[:2]
        pixel_variance = np.var(image)
        
        # Simple fake detection logic based on image characteristics
        confidence = min(0.95, max(0.1, (pixel_variance / 10000.0)))
        is_fake = confidence > 0.5
        
        return {
            'is_fake': is_fake,
            'confidence': confidence,
            'processing_time': np.random.uniform(0.1, 0.3)
        }
    
    def analyze(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Analysis method for frequency and face analyzers"""
        return self.predict(image)

class MockOpenAIDetector:
    """Mock OpenAI detector"""
    
    def analyze_image_bytes(self, file_data: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """Mock OpenAI analysis"""
        # Return None to indicate OpenAI is not available (as expected)
        return None