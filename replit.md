# Deep Fake Detection System - Replit Setup

## Project Overview
This is a Deep Fake Detection System originally designed with production-grade AI models (MesoNet, Xception, etc.) for analyzing images and videos to detect deep fakes. Due to disk space limitations in the Replit environment, a lite version (`app_lite.py`) has been created that provides basic analysis functionality.

## Current Setup
- **Main Application**: `app_lite.py` - Demo version with basic image/video analysis
- **Framework**: Streamlit web application  
- **Python Version**: 3.11.13
- **Host**: 0.0.0.0 (configured for Replit proxy)
- **Port**: 5000
- **Status**: ✅ Working

## Features Available (Lite Version)
- Basic image and video upload
- Metadata extraction from images
- Image property analysis (dimensions, color distribution, brightness)
- Basic face detection using OpenCV Haar cascades
- Video frame analysis and temporal properties
- Interactive dashboard with charts and metrics

## Dependencies Installed
- streamlit>=1.49.1
- numpy>=1.24.0
- opencv-python-headless>=4.8.0 (to avoid GUI dependencies)
- pandas>=2.0.0
- pillow>=10.0.0
- plotly>=5.15.0

## Configuration Files
- `.streamlit/config.toml` - Streamlit configuration for Replit environment
- `pyproject.toml` - Python project dependencies (reduced for space)

## Known Limitations
- **Disk Space**: TensorFlow and heavy AI models cannot be installed due to disk quota limitations
- **AI Models**: MesoNet and Xception models are not available in this version
- **Face Detection**: Using basic OpenCV instead of dlib for facial landmark detection
- **Deep Fake Detection**: Demo confidence scoring only - not actual AI detection

## Original Features (Not Available)
- MesoNet neural network deep fake detection
- Xception-based deep fake classification  
- Advanced facial landmark detection with dlib
- Frequency domain analysis for forgery detection
- Temporal consistency analysis across video frames
- Production-grade confidence scoring

## Workflow Configuration
- **Name**: Streamlit App
- **Command**: `streamlit run app_lite.py`
- **Output**: Web application accessible via Replit's web view

## Deployment Ready
The application is configured for Replit's deployment system with proper host and port settings.

## Recent Changes
- 2025-09-22: Initial setup with disk space optimizations
- Created lite version due to TensorFlow installation constraints
- Configured Streamlit for Replit proxy environment
- Set up basic workflow and deployment preparation