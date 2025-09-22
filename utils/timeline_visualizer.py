"""
Timeline visualization for video frame-by-frame deep fake analysis.
Creates interactive visualizations showing detection results over time.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Tuple, Optional
import cv2
import base64
from io import BytesIO

class TimelineVisualizer:
    """
    Creates interactive timeline visualizations for video analysis results.
    """
    
    def __init__(self):
        self.colors = {
            'real': '#2ecc71',
            'fake': '#e74c3c', 
            'uncertain': '#f39c12',
            'background': '#ecf0f1',
            'text': '#2c3e50'
        }
    
    def create_detection_timeline(self, frame_results: List[Dict], video_metadata: Dict) -> go.Figure:
        """
        Create an interactive timeline showing detection results for each frame.
        
        Args:
            frame_results: List of detection results for each frame
            video_metadata: Video metadata including fps, duration, etc.
            
        Returns:
            Plotly figure with timeline visualization
        """
        if not frame_results:
            return self._create_empty_timeline()
        
        # Prepare data for visualization
        timeline_data = self._prepare_timeline_data(frame_results, video_metadata)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=[
                'Deep Fake Detection Confidence',
                'Model Agreement Score', 
                'Quality Metrics'
            ],
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Add detection confidence timeline
        self._add_confidence_timeline(fig, timeline_data, row=1)
        
        # Add model agreement timeline
        self._add_agreement_timeline(fig, timeline_data, row=2)
        
        # Add quality metrics timeline
        self._add_quality_timeline(fig, timeline_data, row=3)
        
        # Update layout
        fig.update_layout(
            title='Video Deep Fake Analysis Timeline',
            xaxis3_title='Time (seconds)',
            height=800,
            showlegend=True,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def create_frame_by_frame_analysis(self, frame_results: List[Dict], 
                                     frames: List[np.ndarray]) -> go.Figure:
        """
        Create detailed frame-by-frame analysis visualization.
        
        Args:
            frame_results: Detection results for each frame
            frames: Video frames as numpy arrays
            
        Returns:
            Plotly figure with frame-by-frame analysis
        """
        if not frame_results:
            return self._create_empty_timeline()
        
        # Create a simple timeline chart instead of complex grid for better performance
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Frame-by-Frame Detection Results", "Detection Confidence Timeline"],
            vertical_spacing=0.15,
            row_heights=[0.3, 0.7]
        )
        
        # Prepare data
        frame_indices = list(range(len(frame_results)))
        confidences = []
        statuses = []
        colors_list = []
        
        for i, result in enumerate(frame_results):
            confidence = self._get_overall_confidence(result)
            status = self._get_detection_status(confidence)
            
            confidences.append(confidence)
            statuses.append(status)
            colors_list.append(self.colors[status])
        
        # Add frame confidence bar chart
        fig.add_trace(
            go.Bar(
                x=frame_indices,
                y=confidences,
                marker=dict(color=colors_list),
                name='Frame Confidence',
                hovertemplate='Frame %{x}<br>Confidence: %{y:.1%}<br><extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add detection timeline
        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=confidences,
                mode='lines+markers',
                marker=dict(
                    color=colors_list,
                    size=8,
                    line=dict(width=2, color='white')
                ),
                line=dict(width=3),
                name='Detection Timeline',
                hovertemplate='Frame %{x}<br>Confidence: %{y:.1%}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Frame-by-Frame Deep Fake Analysis',
            height=600,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title="Frame Number", row=1, col=1)
        fig.update_yaxes(title="Deep Fake Probability", row=1, col=1)
        fig.update_xaxes(title="Frame Number", row=2, col=1)
        fig.update_yaxes(title="Deep Fake Probability", row=2, col=1)
        
        # Add timeline at the bottom
        timeline_data = self._prepare_timeline_data(frame_results, {'fps': 30})  # Default fps
        x_vals = timeline_data['timestamps']
        y_vals = timeline_data['confidence_scores']
        colors_list = [self.colors[status] for status in timeline_data['status']]
        
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                marker=dict(
                    color=colors_list,
                    size=8,
                    line=dict(width=2, color='white')
                ),
                line=dict(width=3),
                name='Detection Confidence',
                hovertemplate='Time: %{x:.2f}s<br>Confidence: %{y:.1%}<extra></extra>'
            ),
            row=rows + 1, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Frame-by-Frame Deep Fake Analysis',
            height=200 * rows + 300,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_temporal_consistency_plot(self, temporal_analysis: Dict) -> go.Figure:
        """
        Create visualization for temporal consistency analysis.
        
        Args:
            temporal_analysis: Results from temporal analysis
            
        Returns:
            Plotly figure showing temporal consistency metrics
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Motion Smoothness',
                'Illumination Consistency', 
                'Texture Correlation',
                'Frame Differences'
            ]
        )
        
        # Motion smoothness
        if 'motion_smoothness' in temporal_analysis:
            motion_data = temporal_analysis['motion_smoothness']
            if 'position_changes' in motion_data:
                positions = np.array(motion_data['position_changes'])
                if len(positions) > 0:
                    x_motion = positions[:, 0]
                    y_motion = positions[:, 1]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(x_motion))),
                            y=np.sqrt(x_motion**2 + y_motion**2),
                            mode='lines',
                            name='Motion Magnitude',
                            line=dict(color=self.colors['fake'])
                        ),
                        row=1, col=1
                    )
        
        # Illumination consistency
        if 'illumination' in temporal_analysis:
            illum_data = temporal_analysis['illumination']
            if 'brightness_values' in illum_data:
                brightness = illum_data['brightness_values']
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(brightness))),
                        y=brightness,
                        mode='lines',
                        name='Brightness',
                        line=dict(color=self.colors['uncertain'])
                    ),
                    row=1, col=2
                )
        
        # Texture correlation
        if 'texture' in temporal_analysis:
            texture_data = temporal_analysis['texture']
            if 'texture_correlation' in texture_data:
                correlation = texture_data['texture_correlation']
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[correlation, correlation],
                        mode='lines',
                        name=f'Correlation: {correlation:.3f}',
                        line=dict(color=self.colors['real'])
                    ),
                    row=2, col=1
                )
        
        # Frame differences
        if 'frame_differences' in temporal_analysis:
            diff_data = temporal_analysis['frame_differences']
            if 'mean_differences' in diff_data:
                differences = diff_data['mean_differences']
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(differences))),
                        y=differences,
                        mode='lines',
                        name='Frame Differences',
                        line=dict(color=self.colors['text'])
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title='Temporal Consistency Analysis',
            height=600,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_confidence_heatmap(self, frame_results: List[Dict], 
                                 video_metadata: Dict) -> go.Figure:
        """
        Create a heatmap showing confidence levels across different detection methods.
        
        Args:
            frame_results: Detection results for each frame
            video_metadata: Video metadata
            
        Returns:
            Plotly figure with confidence heatmap
        """
        if not frame_results:
            return self._create_empty_timeline()
        
        # Prepare heatmap data
        methods = ['MesoNet', 'Xception', 'Facial Analysis', 'Frequency Analysis']
        confidence_matrix = []
        timestamps = []
        
        fps = video_metadata.get('fps', 30)
        
        for i, result in enumerate(frame_results):
            timestamp = i / fps
            timestamps.append(timestamp)
            
            row = []
            # MesoNet confidence
            meso_conf = result.get('mesonet', {}).get('probability', 0.0)
            row.append(meso_conf)
            
            # Xception confidence  
            xception_conf = result.get('xception', {}).get('probability', 0.0)
            row.append(xception_conf)
            
            # Facial analysis (calculate real confidence score)
            facial_conf = 0.5  # Default neutral
            if 'facial_analysis' in result:
                facial_data = result['facial_analysis']
                score_components = []
                
                # Symmetry analysis
                if 'geometry' in facial_data:
                    geometry = facial_data['geometry']
                    if 'symmetry_score' in geometry:
                        symmetry = geometry['symmetry_score']
                        # Lower symmetry = higher deepfake probability
                        symmetry_score = max(0, (1.0 - symmetry) * 1.5)
                        score_components.append(min(1.0, symmetry_score))
                    
                    # Anomalies
                    if 'anomalies' in geometry:
                        anomaly_count = len(geometry['anomalies'])
                        anomaly_score = min(1.0, anomaly_count * 0.3)
                        score_components.append(anomaly_score)
                
                # Quality inconsistencies
                if 'quality' in facial_data:
                    quality = facial_data['quality']
                    quality_anomalies = 0
                    if 'sharpness' in quality:
                        sharpness = quality['sharpness']
                        if sharpness < 10 or sharpness > 1000:
                            quality_anomalies += 1
                    if 'contrast' in quality:
                        contrast = quality['contrast']
                        if contrast < 10 or contrast > 100:
                            quality_anomalies += 1
                    
                    if quality_anomalies > 0:
                        quality_score = min(1.0, quality_anomalies * 0.3)
                        score_components.append(quality_score)
                
                if score_components:
                    facial_conf = np.mean(score_components)
            
            row.append(facial_conf)
            
            # Frequency analysis (extract real confidence if available)
            freq_conf = 0.5  # Default neutral
            if 'frequency_analysis' in result:
                freq_data = result['frequency_analysis']
                # Calculate frequency-based confidence from actual data
                freq_score = 0.0
                score_count = 0
                
                if 'compression_artifacts' in freq_data:
                    comp_data = freq_data['compression_artifacts']
                    if 'blocking_artifacts' in comp_data:
                        blocking = comp_data['blocking_artifacts']
                        if 'blocking_score' in blocking:
                            freq_score += min(1.0, blocking['blocking_score'] * 2.0)
                            score_count += 1
                
                if 'spectral_irregularities' in freq_data:
                    irreg_data = freq_data['spectral_irregularities']
                    irregularity_score = 0
                    if 'spectral_anomalies' in irreg_data:
                        irregularity_score += min(0.5, irreg_data['spectral_anomalies'] * 0.1)
                    if 'artificial_peaks' in irreg_data:
                        irregularity_score += min(0.3, irreg_data['artificial_peaks'] * 0.1)
                    if irregularity_score > 0:
                        freq_score += min(1.0, irregularity_score)
                        score_count += 1
                
                if score_count > 0:
                    freq_conf = freq_score / score_count
            
            row.append(freq_conf)
            
            confidence_matrix.append(row)
        
        confidence_matrix = np.array(confidence_matrix).T
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=confidence_matrix,
            x=timestamps,
            y=methods,
            colorscale='RdYlBu_r',
            zmid=0.5,
            colorbar=dict(
                title="Deep Fake Probability",
                titleside="right"
            ),
            hoverongaps=False,
            hovertemplate='Method: %{y}<br>Time: %{x:.2f}s<br>Confidence: %{z:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Detection Confidence Heatmap Across Methods',
            xaxis_title='Time (seconds)',
            yaxis_title='Detection Method',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def _prepare_timeline_data(self, frame_results: List[Dict], 
                              video_metadata: Dict) -> Dict:
        """Prepare data for timeline visualization."""
        fps = video_metadata.get('fps', 30)
        
        timestamps = []
        confidence_scores = []
        model_agreements = []
        quality_scores = []
        status_list = []
        
        for i, result in enumerate(frame_results):
            timestamp = i / fps
            timestamps.append(timestamp)
            
            # Overall confidence
            confidence = self._get_overall_confidence(result)
            confidence_scores.append(confidence)
            
            # Model agreement
            agreement = self._calculate_model_agreement(result)
            model_agreements.append(agreement)
            
            # Quality score
            quality = self._calculate_quality_score(result)
            quality_scores.append(quality)
            
            # Status
            status = self._get_detection_status(confidence)
            status_list.append(status)
        
        return {
            'timestamps': timestamps,
            'confidence_scores': confidence_scores,
            'model_agreements': model_agreements, 
            'quality_scores': quality_scores,
            'status': status_list
        }
    
    def _get_overall_confidence(self, result: Dict) -> float:
        """Calculate overall confidence from detection result."""
        confidences = []
        
        # MesoNet
        if 'mesonet' in result and 'probability' in result['mesonet']:
            confidences.append(result['mesonet']['probability'])
        
        # Xception
        if 'xception' in result and 'probability' in result['xception']:
            confidences.append(result['xception']['probability'])
        
        if confidences:
            return np.mean(confidences)
        return 0.5
    
    def _calculate_model_agreement(self, result: Dict) -> float:
        """Calculate agreement between different models."""
        confidences = []
        
        if 'mesonet' in result and 'probability' in result['mesonet']:
            confidences.append(result['mesonet']['probability'])
        
        if 'xception' in result and 'probability' in result['xception']:
            confidences.append(result['xception']['probability'])
        
        if len(confidences) >= 2:
            # Calculate variance - lower variance means higher agreement
            variance = np.var(confidences)
            return max(0.0, 1.0 - float(variance) * 4)  # Scale variance to agreement
        
        return 1.0  # Perfect agreement if only one model
    
    def _calculate_quality_score(self, result: Dict) -> float:
        """Calculate quality score from result."""
        # Placeholder - could be enhanced with actual quality metrics
        return 0.8
    
    def _get_detection_status(self, confidence: float, threshold: float = 0.5) -> str:
        """Get detection status from confidence score."""
        if confidence > threshold + 0.2:
            return 'fake'
        elif confidence < threshold - 0.2:
            return 'real'
        else:
            return 'uncertain'
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 string for display."""
        # Resize frame for thumbnail
        h, w = frame.shape[:2]
        if w > 150:
            scale = 150 / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        
        # Convert to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer).decode()
        return f"data:image/jpeg;base64,{img_str}"
    
    def _add_confidence_timeline(self, fig: go.Figure, data: Dict, row: int):
        """Add confidence timeline to figure."""
        x_vals = data['timestamps']
        y_vals = data['confidence_scores']
        colors_list = [self.colors[status] for status in data['status']]
        
        # Add confidence line
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                marker=dict(
                    color=colors_list,
                    size=8,
                    line=dict(width=2, color='white')
                ),
                line=dict(width=3, color=self.colors['text']),
                name='Detection Confidence',
                hovertemplate='Time: %{x:.2f}s<br>Confidence: %{y:.1%}<extra></extra>'
            ),
            row=row, col=1
        )
        
        # Add threshold lines
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                     annotation_text="Threshold")
    
    def _add_agreement_timeline(self, fig: go.Figure, data: Dict, row: int):
        """Add model agreement timeline to figure."""
        fig.add_trace(
            go.Scatter(
                x=data['timestamps'],
                y=data['model_agreements'],
                mode='lines',
                line=dict(color=self.colors['uncertain'], width=2),
                name='Model Agreement',
                hovertemplate='Time: %{x:.2f}s<br>Agreement: %{y:.1%}<extra></extra>'
            ),
            row=row, col=1
        )
    
    def _add_quality_timeline(self, fig: go.Figure, data: Dict, row: int):
        """Add quality metrics timeline to figure."""
        fig.add_trace(
            go.Scatter(
                x=data['timestamps'],
                y=data['quality_scores'],
                mode='lines',
                line=dict(color=self.colors['real'], width=2),
                name='Quality Score',
                hovertemplate='Time: %{x:.2f}s<br>Quality: %{y:.1%}<extra></extra>'
            ),
            row=row, col=1
        )
    
    def _create_empty_timeline(self) -> go.Figure:
        """Create empty timeline for cases with no data."""
        fig = go.Figure()
        fig.add_annotation(
            text="No timeline data available",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color=self.colors['text'])
        )
        fig.update_layout(
            title="Timeline Analysis",
            height=400,
            template='plotly_white'
        )
        return fig
    
    def export_timeline_data(self, frame_results: List[Dict], 
                           video_metadata: Dict) -> pd.DataFrame:
        """
        Export timeline data as pandas DataFrame for analysis.
        
        Args:
            frame_results: Detection results for each frame
            video_metadata: Video metadata
            
        Returns:
            DataFrame with timeline analysis data
        """
        timeline_data = self._prepare_timeline_data(frame_results, video_metadata)
        
        df = pd.DataFrame({
            'timestamp': timeline_data['timestamps'],
            'confidence_score': timeline_data['confidence_scores'],
            'model_agreement': timeline_data['model_agreements'],
            'quality_score': timeline_data['quality_scores'],
            'detection_status': timeline_data['status']
        })
        
        return df