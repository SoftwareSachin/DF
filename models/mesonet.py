"""
MesoNet implementation for deep fake detection.
Based on the paper "MesoNet: a Compact Facial Video Forgery Detection Network".
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
from typing import Tuple, List
import os

class MesoNet:
    """
    MesoNet model for deep fake detection.
    Implements the architecture from the original paper.
    """
    
    def __init__(self, input_size: Tuple[int, int] = (256, 256)):
        self.input_size = input_size
        self.model = None
        self.build_model()
    
    def build_model(self):
        """Build the MesoNet architecture."""
        inputs = keras.Input(shape=(*self.input_size, 3))
        
        # First convolutional block
        x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        
        # Second convolutional block
        x = layers.Conv2D(8, (5, 5), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        
        # Third convolutional block
        x = layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        
        # Fourth convolutional block
        x = layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(4, 4), padding='same')(x)
        
        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def load_weights(self, weights_path: str):
        """Load pre-trained weights."""
        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)
        else:
            print(f"Warning: Weights file {weights_path} not found. Using randomly initialized weights.")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for MesoNet input."""
        # Resize image
        processed = cv2.resize(image, self.input_size)
        
        # Normalize pixel values
        processed = processed.astype(np.float32) / 255.0
        
        # Add batch dimension
        processed = np.expand_dims(processed, axis=0)
        
        return processed
    
    def predict(self, image: np.ndarray) -> Tuple[float, dict]:
        """
        Predict if an image is a deepfake.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (probability, analysis_details)
        """
        processed_image = self.preprocess_image(image)
        
        # Get prediction
        prediction = self.model.predict(processed_image, verbose=0)[0][0]
        
        # Get intermediate layer outputs for analysis
        layer_outputs = self._get_layer_activations(processed_image)
        
        analysis_details = {
            'mesonet_score': float(prediction),
            'feature_maps': self._analyze_feature_maps(layer_outputs),
            'activation_patterns': self._analyze_activation_patterns(layer_outputs)
        }
        
        return float(prediction), analysis_details
    
    def _get_layer_activations(self, image: np.ndarray) -> List[np.ndarray]:
        """Get activations from intermediate layers."""
        layer_names = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3']
        outputs = [self.model.get_layer(name).output for name in layer_names if name in [layer.name for layer in self.model.layers]]
        
        if outputs:
            activation_model = keras.Model(inputs=self.model.input, outputs=outputs)
            activations = activation_model.predict(image, verbose=0)
            return activations if isinstance(activations, list) else [activations]
        
        return []
    
    def _analyze_feature_maps(self, activations: List[np.ndarray]) -> dict:
        """Analyze feature map characteristics."""
        if not activations:
            return {}
        
        analysis = {}
        for i, activation in enumerate(activations):
            # Calculate statistics for each layer
            mean_activation = np.mean(activation)
            std_activation = np.std(activation)
            max_activation = np.max(activation)
            
            analysis[f'layer_{i}'] = {
                'mean_activation': float(mean_activation),
                'std_activation': float(std_activation),
                'max_activation': float(max_activation),
                'sparsity': float(np.mean(activation == 0))
            }
        
        return analysis
    
    def _analyze_activation_patterns(self, activations: List[np.ndarray]) -> dict:
        """Analyze activation patterns for anomaly detection."""
        if not activations:
            return {}
        
        patterns = {}
        for i, activation in enumerate(activations):
            # Flatten activation for analysis
            flat_activation = activation.flatten()
            
            # Calculate pattern metrics
            entropy = self._calculate_entropy(flat_activation)
            uniformity = self._calculate_uniformity(flat_activation)
            
            patterns[f'layer_{i}'] = {
                'entropy': float(entropy),
                'uniformity': float(uniformity)
            }
        
        return patterns
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate entropy of activation data."""
        # Discretize data for entropy calculation
        hist, _ = np.histogram(data, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    def _calculate_uniformity(self, data: np.ndarray) -> float:
        """Calculate uniformity measure of activations."""
        # Calculate coefficient of variation
        if np.std(data) == 0:
            return 0.0
        return np.std(data) / (np.mean(np.abs(data)) + 1e-10)


class MesoNet4(MesoNet):
    """
    MesoNet-4 variant with 4 convolutional layers.
    """
    
    def build_model(self):
        """Build the MesoNet-4 architecture."""
        inputs = keras.Input(shape=(*self.input_size, 3))
        
        # Conv Block 1
        x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        
        # Conv Block 2
        x = layers.Conv2D(8, (5, 5), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        
        # Conv Block 3
        x = layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        
        # Conv Block 4
        x = layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(4, 4), padding='same')(x)
        
        # Dense layers
        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
