"""
Deep Fake Detection System - Streamlit Web Application (Lite Version)
Demo version for Replit with basic image/video analysis without heavy AI models.
"""
import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from typing import Dict, List, Tuple, Optional
import time
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from PIL import Image
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Deep Fake Detection System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: #f8f9fa;
        padding: 1rem;
        border: 1px solid #dee2e6;
        margin-bottom: 2rem;
    }
    .detection-result {
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        background: #ffffff;
    }
    .analysis-section {
        background: #ffffff;
        padding: 1rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DeepFakeDetectionDemo:
    """Demo version of the deep fake detection system."""
    
    def __init__(self):
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    def render_header(self):
        """Render the application header."""
        st.markdown("""
        <div class="main-header">
            <h1 style="color: #343a40; margin: 0;">Deep Fake Detection System</h1>
            <p style="color: #6c757d; margin: 0;">
                Image and video analysis platform
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        st.sidebar.title("Analysis Settings")
        
        # Analysis options
        st.sidebar.subheader("Configuration")
        analyze_metadata = st.sidebar.checkbox("Metadata Analysis", value=True)
        analyze_properties = st.sidebar.checkbox("Image Properties", value=True)
        basic_face_detection = st.sidebar.checkbox("Face Detection", value=True)
        
        return {
            'analyze_metadata': analyze_metadata,
            'analyze_properties': analyze_properties,
            'basic_face_detection': basic_face_detection
        }
    
    def detect_faces_opencv(self, image):
        """Basic face detection using OpenCV Haar cascades."""
        try:
            # Load OpenCV's pre-trained Haar cascade for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return faces.tolist() if len(faces) > 0 else []
        except Exception as e:
            return []
    
    def analyze_image_properties(self, image):
        """Analyze basic image properties."""
        properties = {
            'dimensions': f"{image.shape[1]}x{image.shape[0]}",
            'channels': image.shape[2] if len(image.shape) == 3 else 1,
            'dtype': str(image.dtype),
            'file_size_estimate': image.nbytes,
            'brightness_mean': float(np.mean(image)),
            'brightness_std': float(np.std(image)),
            'color_distribution': {
                'red_mean': float(np.mean(image[:,:,2])) if len(image.shape) == 3 else 0,
                'green_mean': float(np.mean(image[:,:,1])) if len(image.shape) == 3 else 0,
                'blue_mean': float(np.mean(image[:,:,0])) if len(image.shape) == 3 else 0,
            }
        }
        return properties
    
    def extract_metadata(self, image_bytes):
        """Extract basic metadata from image bytes."""
        try:
            # Convert bytes to PIL Image to get basic info
            pil_image = Image.open(BytesIO(image_bytes))
            
            metadata = {
                'format': pil_image.format,
                'mode': pil_image.mode,
                'size': pil_image.size,
                'has_exif': hasattr(pil_image, '_getexif') and pil_image._getexif() is not None
            }
            
            # Try to get EXIF data
            try:
                if hasattr(pil_image, '_getexif'):
                    exif = pil_image._getexif()
                    if exif:
                        metadata['exif_keys'] = list(exif.keys())[:10]  # First 10 keys
            except:
                pass
                
            return metadata
        except Exception as e:
            return {'error': str(e)}
    
    def process_image(self, uploaded_file, settings: Dict) -> Dict:
        """Process uploaded image for basic analysis."""
        try:
            # Load image
            image_bytes = uploaded_file.read()
            
            # Convert to OpenCV format
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {'error': 'Could not decode image file'}
            
            results = {
                'image_shape': image.shape,
                'analysis': {}
            }
            
            # Extract metadata
            if settings['analyze_metadata']:
                metadata = self.extract_metadata(image_bytes)
                results['metadata'] = metadata
            
            # Analyze image properties
            if settings['analyze_properties']:
                properties = self.analyze_image_properties(image)
                results['properties'] = properties
            
            # Basic face detection
            if settings['basic_face_detection']:
                faces = self.detect_faces_opencv(image)
                results['faces'] = {
                    'count': len(faces),
                    'bounding_boxes': faces
                }
            
            # Generate demo confidence score
            demo_score = self._generate_demo_score(results)
            results['demo_analysis'] = demo_score
            
            return results
            
        except Exception as e:
            return {'error': f'Error processing image: {str(e)}'}
    
    def process_video(self, uploaded_file, settings: Dict) -> Dict:
        """Process uploaded video for basic analysis."""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # Open video file
            cap = cv2.VideoCapture(tmp_file_path)
            
            if not cap.isOpened():
                os.unlink(tmp_file_path)
                return {'error': 'Could not open video file'}
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            results = {
                'video_properties': {
                    'fps': fps,
                    'frame_count': frame_count,
                    'duration': duration,
                    'resolution': f"{width}x{height}"
                },
                'analysis': {}
            }
            
            # Analyze sample frames
            sample_frames = min(5, frame_count)
            frame_interval = max(1, frame_count // sample_frames)
            
            faces_per_frame = []
            frame_properties = []
            
            for i in range(0, frame_count, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret:
                    # Basic face detection
                    if settings['basic_face_detection']:
                        faces = self.detect_faces_opencv(frame)
                        faces_per_frame.append(len(faces))
                    
                    # Frame properties
                    if settings['analyze_properties']:
                        props = self.analyze_image_properties(frame)
                        frame_properties.append(props['brightness_mean'])
            
            cap.release()
            os.unlink(tmp_file_path)
            
            if faces_per_frame:
                results['faces_analysis'] = {
                    'avg_faces_per_frame': np.mean(faces_per_frame),
                    'max_faces_detected': max(faces_per_frame),
                    'frames_with_faces': sum(1 for f in faces_per_frame if f > 0)
                }
            
            if frame_properties:
                results['temporal_analysis'] = {
                    'brightness_variation': np.std(frame_properties),
                    'avg_brightness': np.mean(frame_properties)
                }
            
            # Generate demo confidence score
            demo_score = self._generate_demo_score(results)
            results['demo_analysis'] = demo_score
            
            return results
            
        except Exception as e:
            return {'error': f'Error processing video: {str(e)}'}
    
    def _generate_demo_score(self, results):
        """Generate a demo confidence score based on available analysis."""
        score_components = []
        
        # Face detection component
        if 'faces' in results and results['faces']['count'] > 0:
            face_score = min(0.8, results['faces']['count'] * 0.2)
            score_components.append(face_score)
        elif 'faces_analysis' in results:
            face_score = min(0.8, results['faces_analysis']['avg_faces_per_frame'] * 0.3)
            score_components.append(face_score)
        
        # Properties component
        if 'properties' in results:
            # Normalize brightness to 0-1 range
            brightness_score = min(1.0, results['properties']['brightness_mean'] / 255.0)
            score_components.append(brightness_score * 0.3)
        
        # Calculate final demo score
        if score_components:
            final_score = np.mean(score_components)
        else:
            final_score = 0.5  # Default neutral score
        
        return {
            'confidence': final_score,
            'components': score_components,
            'message': 'Analysis completed using basic detection methods.'
        }
    
    def render_results(self, results: Dict, file_type: str):
        """Render analysis results."""
        if 'error' in results:
            st.error(f"Error: {results['error']}")
            return
        
        # Main demo result
        demo_analysis = results.get('demo_analysis', {})
        confidence = demo_analysis.get('confidence', 0.0)
        
        # Display main result
        st.markdown(f"""
        <div class="detection-result">
            <h2>Analysis Complete</h2>
            <h3>Confidence Score: {confidence:.1%}</h3>
            <p>{demo_analysis.get('message', 'Analysis completed successfully')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different analyses
        if file_type == "image":
            tabs = st.tabs(["Overview", "Image Properties", "Face Analysis", "Metadata"])
        else:
            tabs = st.tabs(["Overview", "Video Properties", "Face Analysis", "Frame Analysis"])
        
        # Overview tab
        with tabs[0]:
            self.render_overview_tab(results, demo_analysis)
        
        # Type-specific tabs
        if file_type == "image":
            with tabs[1]:
                self.render_image_properties_tab(results)
            
            with tabs[2]:
                self.render_face_analysis_tab(results)
            
            with tabs[3]:
                self.render_metadata_tab(results)
        else:
            with tabs[1]:
                self.render_video_properties_tab(results)
            
            with tabs[2]:
                self.render_video_face_analysis_tab(results)
            
            with tabs[3]:
                self.render_frame_analysis_tab(results)
    
    def render_overview_tab(self, results: Dict, demo_analysis: Dict):
        """Render the overview tab."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Confidence Score", f"{demo_analysis.get('confidence', 0):.1%}")
        
        with col2:
            analysis_count = len([k for k in ['faces', 'properties', 'metadata'] if k in results])
            st.metric("Analysis Types", analysis_count)
        
        with col3:
            faces_detected = results.get('faces', {}).get('count', 0)
            if 'faces_analysis' in results:
                faces_detected = int(results['faces_analysis'].get('avg_faces_per_frame', 0))
            st.metric("Faces Found", faces_detected)
        
        # Component breakdown
        st.subheader("Analysis Components")
        components = demo_analysis.get('components', [])
        if components:
            fig = px.bar(
                x=[f"Component {i+1}" for i in range(len(components))],
                y=components,
                title="Analysis Component Scores"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_image_properties_tab(self, results: Dict):
        """Render image properties analysis."""
        if 'properties' not in results:
            st.info("Image properties analysis was not performed.")
            return
        
        properties = results['properties']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Properties")
            st.write(f"**Dimensions:** {properties['dimensions']}")
            st.write(f"**Channels:** {properties['channels']}")
            st.write(f"**Data Type:** {properties['dtype']}")
            st.write(f"**Estimated Size:** {properties['file_size_estimate']:,} bytes")
        
        with col2:
            st.subheader("Color Analysis")
            st.write(f"**Mean Brightness:** {properties['brightness_mean']:.2f}")
            st.write(f"**Brightness Std:** {properties['brightness_std']:.2f}")
            
            if 'color_distribution' in properties:
                color_data = properties['color_distribution']
                colors = ['Red', 'Green', 'Blue']
                values = [color_data['red_mean'], color_data['green_mean'], color_data['blue_mean']]
                
                fig = px.bar(x=colors, y=values, title="Average Color Channel Values")
                st.plotly_chart(fig, use_container_width=True)
    
    def render_face_analysis_tab(self, results: Dict):
        """Render face analysis results."""
        if 'faces' not in results:
            st.info("Face analysis was not performed.")
            return
        
        faces = results['faces']
        
        st.subheader("Face Detection Results")
        st.write(f"**Faces Detected:** {faces['count']}")
        
        if faces['count'] > 0:
            st.write("**Bounding Boxes:**")
            face_df = pd.DataFrame(faces['bounding_boxes'], columns=['X', 'Y', 'Width', 'Height'])
            st.dataframe(face_df)
            
            # Visualize face sizes
            if len(face_df) > 1:
                face_df['Area'] = face_df['Width'] * face_df['Height']
                fig = px.bar(face_df, y='Area', title="Face Areas Detected")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No faces detected in the image.")
    
    def render_metadata_tab(self, results: Dict):
        """Render metadata information."""
        if 'metadata' not in results:
            st.info("Metadata analysis was not performed.")
            return
        
        metadata = results['metadata']
        
        st.subheader("Image Metadata")
        
        for key, value in metadata.items():
            if key != 'error':
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        if 'error' in metadata:
            st.warning(f"Metadata extraction error: {metadata['error']}")
    
    def render_video_properties_tab(self, results: Dict):
        """Render video properties."""
        if 'video_properties' not in results:
            st.info("Video properties analysis was not performed.")
            return
        
        props = results['video_properties']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Video Properties")
            st.write(f"**FPS:** {props['fps']:.2f}")
            st.write(f"**Total Frames:** {props['frame_count']:,}")
            st.write(f"**Duration:** {props['duration']:.2f} seconds")
        
        with col2:
            st.subheader("Video Dimensions")
            st.write(f"**Resolution:** {props['resolution']}")
    
    def render_video_face_analysis_tab(self, results: Dict):
        """Render video face analysis."""
        if 'faces_analysis' not in results:
            st.info("Video face analysis was not performed.")
            return
        
        face_analysis = results['faces_analysis']
        
        st.subheader("Video Face Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Faces/Frame", f"{face_analysis['avg_faces_per_frame']:.2f}")
        
        with col2:
            st.metric("Max Faces Found", face_analysis['max_faces_detected'])
        
        with col3:
            st.metric("Frames with Faces", face_analysis['frames_with_faces'])
    
    def render_frame_analysis_tab(self, results: Dict):
        """Render frame-by-frame analysis."""
        if 'temporal_analysis' not in results:
            st.info("Frame analysis was not performed.")
            return
        
        temporal = results['temporal_analysis']
        
        st.subheader("Temporal Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Brightness Variation", f"{temporal['brightness_variation']:.2f}")
        
        with col2:
            st.metric("Average Brightness", f"{temporal['avg_brightness']:.2f}")
    
    def run(self):
        """Main application runner."""
        self.render_header()
        
        # Sidebar settings
        settings = self.render_sidebar()
        
        # Main content
        st.subheader("File Upload")
        
        tab1, tab2 = st.tabs(["Image Analysis", "Video Analysis"])
        
        with tab1:
            uploaded_image = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                help="Upload an image file for analysis"
            )
            
            if uploaded_image is not None:
                # Display uploaded image
                st.image(uploaded_image, caption=f"Uploaded: {uploaded_image.name}", use_container_width=True)
                
                # Process image
                with st.spinner("Analyzing image..."):
                    results = self.process_image(uploaded_image, settings)
                
                # Display results
                self.render_results(results, "image")
        
        with tab2:
            uploaded_video = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
                help="Upload a video file for analysis"
            )
            
            if uploaded_video is not None:
                # Process video
                with st.spinner("Analyzing video..."):
                    results = self.process_video(uploaded_video, settings)
                
                # Display results
                self.render_results(results, "video")

# Run the application
if __name__ == "__main__":
    app = DeepFakeDetectionDemo()
    app.run()