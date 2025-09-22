"""
Confidence scoring system for deep fake detection.
Combines multiple detection methods to provide a unified confidence score.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class ConfidenceScorer:
    """
    Combines multiple detection methods to calculate a unified confidence score.
    """
    
    def __init__(self):
        # Default weights for different detection methods
        self.default_weights = {
            'mesonet': 0.25,
            'xception': 0.30,
            'facial_analysis': 0.20,
            'frequency_analysis': 0.15,
            'temporal_analysis': 0.10
        }
        
        # Thresholds for different confidence levels
        self.confidence_thresholds = {
            'very_high': 0.90,
            'high': 0.75,
            'medium': 0.60,
            'low': 0.45,
            'very_low': 0.30
        }
    
    def calculate_confidence(self, detections: Dict, deepfake_threshold: float = 0.5) -> Dict:
        """
        Calculate unified confidence score from multiple detection methods.
        
        Args:
            detections: Dictionary containing results from different detection methods
            deepfake_threshold: Threshold for classifying as deep fake
            
        Returns:
            Dictionary containing confidence analysis
        """
        confidence_result = {
            'final_confidence': 0.0,
            'is_deepfake': False,
            'confidence_level': 'unknown',
            'breakdown': {},
            'reliability_factors': {},
            'warnings': []
        }
        
        # Calculate individual method scores
        method_scores = self._calculate_method_scores(detections)
        
        # Calculate weighted confidence
        weighted_scores = self._calculate_weighted_scores(method_scores)
        
        # Apply reliability adjustments
        reliability_factors = self._analyze_reliability_factors(detections)
        
        # Calculate final confidence
        final_confidence = self._calculate_final_confidence(
            weighted_scores, 
            reliability_factors
        )
        
        # Determine if deepfake
        is_deepfake = final_confidence > deepfake_threshold
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(final_confidence)
        
        # Generate warnings
        warnings = self._generate_warnings(detections, reliability_factors)
        
        confidence_result.update({
            'final_confidence': final_confidence,
            'is_deepfake': is_deepfake,
            'confidence_level': confidence_level,
            'breakdown': weighted_scores,
            'reliability_factors': reliability_factors,
            'warnings': warnings,
            'method_scores': method_scores
        })
        
        return confidence_result
    
    def _calculate_method_scores(self, detections: Dict) -> Dict:
        """Calculate normalized scores for each detection method."""
        method_scores = {}
        
        # MesoNet score
        if 'mesonet' in detections and 'error' not in detections['mesonet']:
            meso_prob = detections['mesonet'].get('probability', 0.0)
            method_scores['mesonet'] = {
                'raw_score': meso_prob,
                'normalized_score': meso_prob,
                'available': True
            }
        else:
            method_scores['mesonet'] = {'available': False}
        
        # Xception score
        if 'xception' in detections and 'error' not in detections['xception']:
            xception_prob = detections['xception'].get('probability', 0.0)
            method_scores['xception'] = {
                'raw_score': xception_prob,
                'normalized_score': xception_prob,
                'available': True
            }
        else:
            method_scores['xception'] = {'available': False}
        
        # Facial analysis score
        if 'facial_analysis' in detections and 'error' not in detections['facial_analysis']:
            facial_score = self._calculate_facial_analysis_score(detections['facial_analysis'])
            method_scores['facial_analysis'] = {
                'raw_score': facial_score,
                'normalized_score': facial_score,
                'available': True
            }
        else:
            method_scores['facial_analysis'] = {'available': False}
        
        # Frequency analysis score
        if 'frequency_analysis' in detections and 'error' not in detections['frequency_analysis']:
            freq_score = self._calculate_frequency_analysis_score(detections['frequency_analysis'])
            method_scores['frequency_analysis'] = {
                'raw_score': freq_score,
                'normalized_score': freq_score,
                'available': True
            }
        else:
            method_scores['frequency_analysis'] = {'available': False}
        
        # Temporal analysis score (for videos)
        if 'temporal_analysis' in detections and 'error' not in detections['temporal_analysis']:
            temporal_score = self._calculate_temporal_analysis_score(detections['temporal_analysis'])
            method_scores['temporal_analysis'] = {
                'raw_score': temporal_score,
                'normalized_score': temporal_score,
                'available': True
            }
        else:
            method_scores['temporal_analysis'] = {'available': False}
        
        return method_scores
    
    def _calculate_facial_analysis_score(self, facial_data: Dict) -> float:
        """Calculate score from facial analysis results."""
        score_components = []
        
        # Geometry analysis
        if 'geometry' in facial_data:
            geometry = facial_data['geometry']
            
            # Symmetry score (lower symmetry = higher deepfake probability)
            if 'symmetry_score' in geometry:
                symmetry = geometry['symmetry_score']
                # Convert to deepfake probability (inverted)
                symmetry_score = max(0, (1.0 - symmetry) * 1.5)  # Amplify the effect
                score_components.append(min(1.0, symmetry_score))
            
            # Anomalies
            if 'anomalies' in geometry:
                anomaly_count = len(geometry['anomalies'])
                # Each anomaly increases deepfake probability
                anomaly_score = min(1.0, anomaly_count * 0.3)
                score_components.append(anomaly_score)
            
            # Proportions analysis
            if 'proportions' in geometry:
                proportions = geometry['proportions']
                prop_anomalies = 0
                
                # Check for unusual proportions
                if 'width_height_ratio' in proportions:
                    ratio = proportions['width_height_ratio']
                    if ratio < 0.6 or ratio > 1.2:
                        prop_anomalies += 1
                
                if 'eye_separation_ratio' in proportions:
                    eye_ratio = proportions['eye_separation_ratio']
                    if eye_ratio < 2.0 or eye_ratio > 4.0:
                        prop_anomalies += 1
                
                if 'nose_mouth_ratio' in proportions:
                    nose_mouth = proportions['nose_mouth_ratio']
                    if nose_mouth < 0.4 or nose_mouth > 0.8:
                        prop_anomalies += 1
                
                prop_score = min(1.0, prop_anomalies * 0.25)
                score_components.append(prop_score)
        
        # Quality analysis
        if 'quality' in facial_data:
            quality = facial_data['quality']
            quality_anomalies = 0
            
            # Check for quality inconsistencies that might indicate manipulation
            if 'sharpness' in quality:
                # Very low or very high sharpness can indicate manipulation
                sharpness = quality['sharpness']
                if sharpness < 10 or sharpness > 1000:
                    quality_anomalies += 1
            
            if 'contrast' in quality:
                # Unusual contrast values
                contrast = quality['contrast']
                if contrast < 10 or contrast > 100:
                    quality_anomalies += 1
            
            quality_score = min(1.0, quality_anomalies * 0.3)
            score_components.append(quality_score)
        
        # Calculate average score
        if score_components:
            return np.mean(score_components)
        else:
            return 0.0
    
    def _calculate_frequency_analysis_score(self, freq_data: Dict) -> float:
        """Calculate score from frequency analysis results."""
        score_components = []
        
        # DCT analysis
        if 'dct_analysis' in freq_data:
            dct_data = freq_data['dct_analysis']
            
            # Quantization artifacts
            if 'quantization_artifacts' in dct_data:
                # High number of quantization artifacts suggests manipulation
                artifacts = dct_data['quantization_artifacts']
                # Normalize by typical number of blocks analyzed
                artifact_score = min(1.0, artifacts / 100.0)
                score_components.append(artifact_score)
            
            # DC coefficient variance
            if 'dc_variance' in dct_data:
                # Very high or low variance can indicate manipulation
                variance = dct_data['dc_variance']
                if variance > 10000 or variance < 100:
                    score_components.append(0.3)
        
        # FFT analysis
        if 'fft_analysis' in freq_data:
            fft_data = freq_data['fft_analysis']
            
            # Spectral anomalies
            spectral_anomalies = 0
            
            if 'spectral_flatness' in fft_data:
                flatness = fft_data['spectral_flatness']
                # Very high flatness (noise-like) or very low (tonal) can indicate manipulation
                if flatness > 0.8 or flatness < 0.1:
                    spectral_anomalies += 1
            
            if 'frequency_peaks' in fft_data:
                peaks_info = fft_data['frequency_peaks']
                if 'num_peaks' in peaks_info and peaks_info['num_peaks'] > 20:
                    spectral_anomalies += 1
            
            if spectral_anomalies > 0:
                score_components.append(min(1.0, spectral_anomalies * 0.4))
        
        # Compression artifacts
        if 'compression_artifacts' in freq_data:
            comp_data = freq_data['compression_artifacts']
            
            if 'blocking_artifacts' in comp_data:
                blocking = comp_data['blocking_artifacts']
                if 'blocking_score' in blocking:
                    # High blocking score indicates heavy compression/recompression
                    blocking_score = min(1.0, blocking['blocking_score'] * 2.0)
                    score_components.append(blocking_score)
            
            if 'compression_ratio_estimate' in comp_data:
                # Very low quality estimates suggest heavy compression
                quality = comp_data['compression_ratio_estimate']
                if quality < 50:
                    score_components.append(0.4)
        
        # Spectral irregularities
        if 'spectral_irregularities' in freq_data:
            irreg_data = freq_data['spectral_irregularities']
            
            irregularity_score = 0
            if 'spectral_anomalies' in irreg_data:
                anomalies = irreg_data['spectral_anomalies']
                irregularity_score += min(0.5, anomalies * 0.1)
            
            if 'frequency_gaps' in irreg_data:
                gaps = irreg_data['frequency_gaps']
                irregularity_score += min(0.3, gaps * 0.15)
            
            if 'artificial_peaks' in irreg_data:
                peaks = irreg_data['artificial_peaks']
                irregularity_score += min(0.2, peaks * 0.1)
            
            if irregularity_score > 0:
                score_components.append(min(1.0, irregularity_score))
        
        # Calculate average score
        if score_components:
            return np.mean(score_components)
        else:
            return 0.0
    
    def _calculate_temporal_analysis_score(self, temporal_data: Dict) -> float:
        """Calculate score from temporal analysis results."""
        score_components = []
        
        # Motion smoothness
        if 'motion_smoothness' in temporal_data:
            motion_data = temporal_data['motion_smoothness']
            
            if 'motion_smoothness_score' in motion_data:
                smoothness = motion_data['motion_smoothness_score']
                # Low smoothness indicates potential manipulation
                smoothness_score = max(0, (1.0 - smoothness))
                score_components.append(smoothness_score)
        
        # Optical flow analysis
        if 'optical_flow' in temporal_data:
            flow_data = temporal_data['optical_flow']
            
            flow_anomalies = 0
            
            if 'anomalous_motion' in flow_data:
                anomalous_count = flow_data['anomalous_motion']
                if anomalous_count > 5:  # Threshold for suspicious motion
                    flow_anomalies += 1
            
            if 'flow_consistency' in flow_data:
                consistency = flow_data['flow_consistency']
                if consistency < 0.5:  # Low consistency
                    flow_anomalies += 1
            
            if flow_anomalies > 0:
                score_components.append(min(1.0, flow_anomalies * 0.4))
        
        # Frame differences
        if 'frame_differences' in temporal_data:
            diff_data = temporal_data['frame_differences']
            
            if 'difference_consistency' in diff_data:
                consistency = diff_data['difference_consistency']
                # Low consistency indicates potential manipulation
                if consistency < 0.6:
                    score_components.append(1.0 - consistency)
        
        # Illumination consistency
        if 'illumination' in temporal_data:
            illum_data = temporal_data['illumination']
            
            if 'illumination_jumps' in illum_data:
                jumps = illum_data['illumination_jumps']
                # Sudden illumination changes are suspicious
                jump_score = min(1.0, jumps * 0.2)
                score_components.append(jump_score)
            
            if 'brightness_variance' in illum_data:
                variance = illum_data['brightness_variance']
                # High variance indicates inconsistent lighting
                if variance > 100:
                    score_components.append(0.3)
        
        # Texture consistency
        if 'texture' in temporal_data:
            texture_data = temporal_data['texture']
            
            if 'texture_correlation' in texture_data:
                correlation = texture_data['texture_correlation']
                # Low correlation indicates texture inconsistencies
                if correlation < 0.7:
                    score_components.append(1.0 - correlation)
        
        # Calculate average score
        if score_components:
            return np.mean(score_components)
        else:
            return 0.0
    
    def _calculate_weighted_scores(self, method_scores: Dict) -> Dict:
        """Calculate weighted scores for each available method."""
        weighted_scores = {}
        total_weight = 0.0
        
        # Calculate weights for available methods
        available_methods = [method for method, data in method_scores.items() 
                           if data.get('available', False)]
        
        if not available_methods:
            return {}
        
        # Redistribute weights for unavailable methods
        method_weights = {}
        base_total_weight = sum(self.default_weights[method] for method in available_methods)
        
        for method in available_methods:
            # Normalize weights to sum to 1.0
            method_weights[method] = self.default_weights[method] / base_total_weight
        
        # Calculate weighted scores
        for method in available_methods:
            score = method_scores[method]['normalized_score']
            weight = method_weights[method]
            weighted_score = score * weight
            
            weighted_scores[method] = {
                'score': score,
                'weight': weight,
                'weighted_score': weighted_score
            }
            
            total_weight += weight
        
        return weighted_scores
    
    def _analyze_reliability_factors(self, detections: Dict) -> Dict:
        """Analyze factors that affect the reliability of the detection."""
        reliability_factors = {
            'model_consensus': 0.0,
            'detection_coverage': 0.0,
            'data_quality': 1.0,
            'consistency_score': 1.0,
            'reliability_multiplier': 1.0
        }
        
        # Model consensus - how much do different models agree
        model_scores = []
        for method in ['mesonet', 'xception']:
            if method in detections and 'error' not in detections[method]:
                prob = detections[method].get('probability', 0.0)
                model_scores.append(prob)
        
        if len(model_scores) >= 2:
            # Calculate variance in model predictions
            score_variance = np.var(model_scores)
            # High variance = low consensus, low variance = high consensus
            consensus = max(0.0, 1.0 - (score_variance * 4))  # Scale variance
            reliability_factors['model_consensus'] = consensus
        
        # Detection coverage - how many detection methods are active
        total_methods = len(self.default_weights)
        active_methods = sum(1 for method, data in detections.items() 
                           if 'error' not in data)
        
        coverage = active_methods / total_methods
        reliability_factors['detection_coverage'] = coverage
        
        # Data quality factors
        quality_issues = 0
        
        # Check for face detection issues
        if 'facial_analysis' in detections:
            facial_data = detections['facial_analysis']
            if 'error' in facial_data:
                quality_issues += 1
            elif 'quality' in facial_data:
                quality = facial_data['quality']
                # Check for very low quality metrics
                if quality.get('sharpness', 100) < 10:
                    quality_issues += 1
                if quality.get('contrast', 50) < 10:
                    quality_issues += 1
        
        # Adjust data quality based on issues
        if quality_issues > 0:
            reliability_factors['data_quality'] = max(0.3, 1.0 - (quality_issues * 0.3))
        
        # Consistency score across different analysis methods
        analysis_scores = []
        for method, data in detections.items():
            if 'error' not in data:
                if method == 'mesonet':
                    analysis_scores.append(data.get('probability', 0.0))
                elif method == 'xception':
                    analysis_scores.append(data.get('probability', 0.0))
                elif method == 'facial_analysis':
                    # Convert facial analysis to a probability-like score
                    facial_score = self._calculate_facial_analysis_score(data)
                    analysis_scores.append(facial_score)
        
        if len(analysis_scores) >= 2:
            # Calculate coefficient of variation
            mean_score = np.mean(analysis_scores)
            if mean_score > 0:
                cv = np.std(analysis_scores) / mean_score
                consistency = max(0.0, 1.0 - cv)  # Lower CV = higher consistency
                reliability_factors['consistency_score'] = consistency
        
        # Calculate overall reliability multiplier
        reliability_multiplier = (
            reliability_factors['model_consensus'] * 0.3 +
            reliability_factors['detection_coverage'] * 0.4 +
            reliability_factors['data_quality'] * 0.2 +
            reliability_factors['consistency_score'] * 0.1
        )
        
        reliability_factors['reliability_multiplier'] = reliability_multiplier
        
        return reliability_factors
    
    def _calculate_final_confidence(self, weighted_scores: Dict, 
                                  reliability_factors: Dict) -> float:
        """Calculate the final confidence score."""
        if not weighted_scores:
            return 0.0
        
        # Sum weighted scores
        base_confidence = sum(score_data['weighted_score'] 
                            for score_data in weighted_scores.values())
        
        # Apply reliability adjustment
        reliability_multiplier = reliability_factors.get('reliability_multiplier', 1.0)
        
        # Ensure reliability multiplier is reasonable
        reliability_multiplier = max(0.1, min(1.5, reliability_multiplier))
        
        # Calculate final confidence
        final_confidence = base_confidence * reliability_multiplier
        
        # Ensure confidence is in valid range [0, 1]
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        return final_confidence
    
    def _determine_confidence_level(self, confidence: float) -> str:
        """Determine the confidence level based on the score."""
        if confidence >= self.confidence_thresholds['very_high']:
            return 'very_high'
        elif confidence >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'medium'
        elif confidence >= self.confidence_thresholds['low']:
            return 'low'
        else:
            return 'very_low'
    
    def _generate_warnings(self, detections: Dict, 
                          reliability_factors: Dict) -> List[str]:
        """Generate warnings about the analysis."""
        warnings = []
        
        # Check for low detection coverage
        coverage = reliability_factors.get('detection_coverage', 1.0)
        if coverage < 0.5:
            warnings.append("Low detection coverage - some analysis methods unavailable")
        
        # Check for model consensus issues
        consensus = reliability_factors.get('model_consensus', 1.0)
        if consensus < 0.6:
            warnings.append("Low model consensus - different AI models disagree significantly")
        
        # Check for data quality issues
        data_quality = reliability_factors.get('data_quality', 1.0)
        if data_quality < 0.7:
            warnings.append("Data quality issues detected - results may be less reliable")
        
        # Check for specific method errors
        error_methods = [method for method, data in detections.items() 
                        if 'error' in data]
        if error_methods:
            method_names = ', '.join(error_methods)
            warnings.append(f"Analysis errors in: {method_names}")
        
        # Check for insufficient face detection
        if 'facial_analysis' in detections:
            facial_data = detections['facial_analysis']
            if 'landmarks_count' in facial_data and facial_data['landmarks_count'] < 50:
                warnings.append("Limited facial landmarks detected - facial analysis may be incomplete")
        
        return warnings
    
    def get_confidence_explanation(self, confidence_result: Dict) -> str:
        """Generate a human-readable explanation of the confidence score."""
        confidence = confidence_result['final_confidence']
        level = confidence_result['confidence_level']
        is_deepfake = confidence_result['is_deepfake']
        
        explanation_parts = []
        
        # Main result
        if is_deepfake:
            explanation_parts.append(f"The analysis indicates this is likely a deep fake with {confidence:.1%} confidence.")
        else:
            explanation_parts.append(f"The analysis suggests this is likely authentic with {(1-confidence):.1%} confidence.")
        
        # Confidence level interpretation
        level_descriptions = {
            'very_high': "Very high confidence - the analysis is highly reliable.",
            'high': "High confidence - the analysis is reliable with strong evidence.",
            'medium': "Medium confidence - the analysis provides reasonable evidence.",
            'low': "Low confidence - the analysis has limited evidence.",
            'very_low': "Very low confidence - the analysis is uncertain."
        }
        
        if level in level_descriptions:
            explanation_parts.append(level_descriptions[level])
        
        # Contributing factors
        breakdown = confidence_result.get('breakdown', {})
        if breakdown:
            top_contributors = sorted(breakdown.items(), 
                                    key=lambda x: x[1]['weighted_score'], 
                                    reverse=True)[:2]
            
            if top_contributors:
                contrib_names = [name.replace('_', ' ').title() for name, _ in top_contributors]
                explanation_parts.append(f"Primary evidence from: {', '.join(contrib_names)}.")
        
        # Warnings
        warnings = confidence_result.get('warnings', [])
        if warnings:
            explanation_parts.append(f"Note: {'; '.join(warnings)}.")
        
        return ' '.join(explanation_parts)
