"""
Real AI-Powered Deep Fake Detection using OpenAI Vision Model
"""
import json
import os
import base64
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import time
import logging

# Using OpenAI GPT-4o vision model for deep fake detection
from openai import OpenAI
from enhanced_detection import AuthenticityAnalyzer

logger = logging.getLogger(__name__)

class OpenAIDeepFakeDetector:
    """Advanced deep fake detection using OpenAI's multimodal vision model"""
    
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found - OpenAI analysis will be disabled")
            self.client = None
        else:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        self.model = "gpt-4o"  # Current OpenAI vision model
        
    def analyze_image_for_deepfake(self, image: np.ndarray) -> Dict:
        """
        Analyze image using OpenAI's advanced vision model for deep fake detection
        """
        start_time = time.time()
        
        # Check if OpenAI client is available
        if not self.client:
            return {
                'model': 'OpenAI GPT-4o Vision',
                'confidence': 0.0,
                'is_fake': False,
                'inference_time': time.time() - start_time,
                'error': 'OpenAI API not available',
                'analysis': 'OpenAI analysis disabled - API key not configured'
            }
        
        try:
            # Convert image to base64
            base64_image = self._image_to_base64(image)
            
            # Construct specialized prompt for deep fake detection
            prompt = self._create_deepfake_analysis_prompt()
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in digital forensics and deep fake detection. Analyze images for signs of AI-generated or manipulated content with high precision."
                    },
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=2048
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            inference_time = time.time() - start_time
            
            # Standardize response format
            return {
                'model': 'OpenAI GPT-4o Vision',
                'confidence': result.get('deepfake_probability', 0.5),
                'is_fake': result.get('is_deepfake', False),
                'inference_time': inference_time,
                'analysis': result.get('detailed_analysis', ''),
                'indicators': result.get('deepfake_indicators', []),
                'authenticity_score': result.get('authenticity_score', 0.5),
                'risk_level': result.get('risk_level', 'medium')
            }
            
        except Exception as e:
            logger.error(f"OpenAI deepfake analysis failed: {e}")
            return {
                'model': 'OpenAI GPT-4o Vision',
                'confidence': 0.0,
                'is_fake': False,
                'inference_time': time.time() - start_time,
                'error': str(e),
                'analysis': 'OpenAI analysis failed - using fallback detection only'
            }
    
    def _create_deepfake_analysis_prompt(self) -> str:
        """Create specialized prompt for deep fake detection"""
        return """
Analyze this image for signs of AI-generated content or deep fake manipulation. Look for:

TECHNICAL INDICATORS:
- Compression artifacts inconsistencies
- Unnatural lighting patterns
- Pixel-level anomalies
- Edge inconsistencies
- Color gradient irregularities

FACIAL ANALYSIS (if face present):
- Asymmetrical facial features
- Unnatural eye movements or blinking patterns
- Inconsistent skin texture
- Teeth and mouth irregularities
- Hair texture and strand patterns
- Neck and jawline continuity

AI GENERATION ARTIFACTS:
- Repetitive patterns typical of GANs
- Smoothing artifacts from neural networks
- Unnatural background blending
- Geometric inconsistencies

AUTHENTICITY MARKERS:
- Natural imperfections that indicate real photography
- Consistent lighting and shadows
- Realistic texture variations
- Proper depth of field
- Camera sensor noise patterns
- Natural color saturation
- Realistic motion blur or focus

ADDITIONAL ANALYSIS:
- Compare image characteristics to known real photo patterns
- Analyze metadata and EXIF data consistency
- Check for natural randomness in pixel patterns
- Evaluate realistic human micro-expressions
- Assess environmental lighting consistency

Provide your analysis in JSON format:
{
  "is_deepfake": boolean,
  "deepfake_probability": float (0.0-1.0),
  "authenticity_score": float (0.0-1.0, where 1.0 is authentic),
  "risk_level": "low|medium|high",
  "deepfake_indicators": ["list of specific indicators found"],
  "authenticity_markers": ["list of authenticity indicators"],
  "detailed_analysis": "detailed explanation of findings",
  "confidence_reasoning": "explanation of confidence level",
  "image_type": "likely source (e.g., 'professional_photo', 'smartphone', 'ai_generated', 'edited')",
  "technical_quality": "assessment of technical aspects",
  "recommendation": "suggested action based on analysis"
}

Be thorough and precise in your analysis.
"""
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string"""
        # Resize image if too large (API limits)
        height, width = image.shape[:2]
        if width > 1024 or height > 1024:
            scale = min(1024/width, 1024/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Convert to JPEG
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Convert to base64
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        return base64_image


class AdvancedComputerVisionDetector:
    """Advanced computer vision techniques for deep fake detection"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def analyze_compression_artifacts(self, image: np.ndarray) -> Dict:
        """Analyze compression artifacts that may indicate manipulation"""
        start_time = time.time()
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # DCT analysis for compression artifacts
            dct = cv2.dct(np.float32(gray))
            
            # Analyze high frequency components
            h, w = dct.shape
            high_freq_region = dct[h//2:, w//2:]
            high_freq_energy = np.mean(np.abs(high_freq_region))
            
            # Analyze block artifacts (8x8 DCT blocks)
            block_artifacts = self._detect_block_artifacts(gray)
            
            # Edge consistency analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Calculate overall suspicion score
            artifact_score = min(1.0, (block_artifacts + high_freq_energy * 0.001) / 2)
            
            inference_time = time.time() - start_time
            
            return {
                'model': 'Computer Vision Artifacts',
                'confidence': artifact_score,
                'is_fake': artifact_score > 0.6,
                'inference_time': inference_time,
                'high_freq_energy': high_freq_energy,
                'block_artifacts': block_artifacts,
                'edge_density': edge_density,
                'analysis': f"Compression artifact analysis completed. Artifact score: {artifact_score:.3f}"
            }
            
        except Exception as e:
            logger.error(f"Compression analysis failed: {e}")
            return {
                'model': 'Computer Vision Artifacts',
                'confidence': 0.5,
                'is_fake': False,
                'inference_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _detect_block_artifacts(self, gray_image: np.ndarray) -> float:
        """Detect 8x8 block artifacts typical in manipulated images"""
        h, w = gray_image.shape
        block_score = 0.0
        block_count = 0
        
        # Analyze 8x8 blocks
        for y in range(0, h - 8, 8):
            for x in range(0, w - 8, 8):
                block = gray_image[y:y+8, x:x+8]
                
                # Calculate block variance
                block_var = np.var(block)
                
                # Check for unnatural uniformity
                if block_var < 10:  # Very uniform block
                    block_score += 1.0
                elif block_var > 1000:  # Very noisy block
                    block_score += 0.5
                
                block_count += 1
        
        return block_score / max(1, block_count) if block_count > 0 else 0.0


class EnsembleRealDeepFakeDetector:
    """Ensemble detector combining multiple real AI approaches"""
    
    def __init__(self):
        self.openai_detector = OpenAIDeepFakeDetector()
        self.cv_detector = AdvancedComputerVisionDetector()
        self.authenticity_analyzer = AuthenticityAnalyzer()
        
        # Weights for ensemble
        self.weights = {
            'openai_vision': 0.5,     # Advanced AI vision
            'cv_artifacts': 0.25,     # Computer vision analysis
            'authenticity': 0.25      # Enhanced authenticity analysis
        }
    
    def predict(self, image: np.ndarray) -> Dict:
        """Comprehensive deep fake prediction using ensemble approach"""
        start_time = time.time()
        
        try:
            # Run all detectors
            openai_result = self.openai_detector.analyze_image_for_deepfake(image)
            cv_result = self.cv_detector.analyze_compression_artifacts(image)
            authenticity_result = self.authenticity_analyzer.analyze_image_authenticity(image)
            
            # Calculate ensemble confidence (convert authenticity to deepfake confidence)
            authenticity_confidence = 1.0 - authenticity_result.get('authenticity_score', 0.5)
            
            ensemble_confidence = (
                openai_result['confidence'] * self.weights['openai_vision'] +
                cv_result['confidence'] * self.weights['cv_artifacts'] +
                authenticity_confidence * self.weights['authenticity']
            )
            
            # Determine final prediction
            is_fake = ensemble_confidence > 0.5
            
            total_time = time.time() - start_time
            
            return {
                'model': 'Real AI Ensemble (OpenAI + CV)',
                'confidence': ensemble_confidence,
                'is_fake': is_fake,
                'inference_time': total_time,
                'ensemble_results': {
                    'openai_vision': openai_result,
                    'computer_vision': cv_result,
                    'authenticity_analysis': authenticity_result
                },
                'risk_assessment': self._assess_risk(ensemble_confidence, openai_result, cv_result, authenticity_result),
                'explanation': self._generate_explanation(openai_result, cv_result, ensemble_confidence, authenticity_result)
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return {
                'model': 'Real AI Ensemble (OpenAI + CV)',
                'confidence': 0.5,
                'is_fake': False,
                'inference_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _assess_risk(self, confidence: float, openai_result: Dict, cv_result: Dict, authenticity_result: Dict = None) -> str:
        """Assess overall risk level"""
        if confidence > 0.8:
            return "HIGH - Strong indicators of manipulation"
        elif confidence > 0.6:
            return "MEDIUM-HIGH - Multiple suspicious indicators"
        elif confidence > 0.4:
            return "MEDIUM - Some concerning features detected"
        elif confidence > 0.2:
            return "LOW-MEDIUM - Minor irregularities found"
        else:
            return "LOW - Appears authentic"
    
    def _generate_explanation(self, openai_result: Dict, cv_result: Dict, confidence: float, authenticity_result: Dict = None) -> str:
        """Generate human-readable explanation"""
        explanation = f"Analysis confidence: {confidence:.1%}\n\n"
        
        if 'analysis' in openai_result:
            explanation += f"AI Vision Analysis: {openai_result['analysis']}\n\n"
        
        explanation += f"Technical Analysis: Found compression artifacts with score {cv_result.get('confidence', 0):.3f}\n"
        
        if confidence > 0.6:
            explanation += "\nWARNING: This image shows significant signs of AI generation or manipulation."
        elif confidence > 0.4:
            explanation += "\nNOTICE: This image has some suspicious characteristics that warrant further investigation."
        else:
            explanation += "\nRESULT: This image appears to be authentic based on current analysis."
        
        return explanation