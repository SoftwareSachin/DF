"""
Xception-based deep fake detector.
Fine-tuned on FaceForensics++ dataset for deep fake detection.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers
import numpy as np
import cv2
from typing import Tuple, Dict
import os

class XceptionDetector:
    """
    Xception-based deep fake detection model.
    """
    
    def __init__(self, input_size: Tuple[int, int] = (299, 299)):
        self.input_size = input_size
        self.model = None
        self.base_model = None
        self.build_model()
    
    def build_model(self):
        """Build Xception-based model for deep fake detection."""
        # Load pre-trained Xception model
        self.base_model = Xception(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.input_size, 3),
            pooling='avg'
        )
        
        # Freeze base model layers initially
        self.base_model.trainable = False
        
        # Add custom classification head
        inputs = keras.Input(shape=(*self.input_size, 3))
        x = self.base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x) if x.shape.rank > 2 else x
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def load_weights(self, weights_path: str):
        """Load fine-tuned weights."""
        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)
        else:
            print(f"Warning: Weights file {weights_path} not found. Using base Xception weights.")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for Xception input."""
        # Resize image
        processed = cv2.resize(image, self.input_size)
        
        # Convert BGR to RGB if needed
        if len(processed.shape) == 3 and processed.shape[2] == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # Normalize for Xception (values between -1 and 1)
        processed = processed.astype(np.float32)
        processed = keras.applications.xception.preprocess_input(processed)
        
        # Add batch dimension
        processed = np.expand_dims(processed, axis=0)
        
        return processed
    
    def predict(self, image: np.ndarray) -> Tuple[float, Dict]:
        """
        Predict if an image is a deepfake using Xception.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (probability, analysis_details)
        """
        processed_image = self.preprocess_image(image)
        
        # Get prediction
        prediction = self.model.predict(processed_image, verbose=0)[0][0]
        
        # Get feature analysis
        features = self._extract_features(processed_image)
        
        analysis_details = {
            'xception_score': float(prediction),
            'feature_analysis': self._analyze_features(features),
            'gradient_analysis': self._analyze_gradients(processed_image),
            'attention_maps': self._generate_attention_maps(processed_image)
        }
        
        return float(prediction), analysis_details
    
    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from the base Xception model."""
        feature_extractor = keras.Model(
            inputs=self.base_model.input,
            outputs=self.base_model.output
        )
        features = feature_extractor.predict(image, verbose=0)
        return features
    
    def _analyze_features(self, features: np.ndarray) -> Dict:
        """Analyze extracted features for anomalies."""
        analysis = {
            'feature_mean': float(np.mean(features)),
            'feature_std': float(np.std(features)),
            'feature_max': float(np.max(features)),
            'feature_min': float(np.min(features)),
            'feature_sparsity': float(np.mean(features == 0)),
            'feature_energy': float(np.sum(features ** 2))
        }
        
        # Calculate feature distribution metrics
        flat_features = features.flatten()
        analysis['feature_skewness'] = float(self._calculate_skewness(flat_features))
        analysis['feature_kurtosis'] = float(self._calculate_kurtosis(flat_features))
        
        return analysis
    
    def _analyze_gradients(self, image: np.ndarray) -> Dict:
        """Analyze gradients for input attribution."""
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = self.model(image)
        
        gradients = tape.gradient(prediction, image)
        
        if gradients is not None:
            grad_magnitude = tf.reduce_mean(tf.abs(gradients))
            grad_variance = tf.reduce_mean(tf.square(gradients - tf.reduce_mean(gradients)))
            
            return {
                'gradient_magnitude': float(grad_magnitude),
                'gradient_variance': float(grad_variance),
                'gradient_norm': float(tf.norm(gradients))
            }
        
        return {'gradient_magnitude': 0.0, 'gradient_variance': 0.0, 'gradient_norm': 0.0}
    
    def _generate_attention_maps(self, image: np.ndarray) -> Dict:
        """Generate attention maps using Grad-CAM."""
        try:
            # Get the last convolutional layer
            last_conv_layer = None
            for layer in reversed(self.base_model.layers):
                if isinstance(layer, layers.Conv2D):
                    last_conv_layer = layer
                    break
            
            if last_conv_layer is None:
                return {'attention_score': 0.0}
            
            # Create model for Grad-CAM
            grad_model = keras.Model(
                inputs=self.model.input,
                outputs=[last_conv_layer.output, self.model.output]
            )
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image)
                class_output = predictions[:, 0]
            
            grads = tape.gradient(class_output, conv_outputs)
            
            if grads is not None:
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_outputs = conv_outputs[0]
                heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
                
                attention_score = float(tf.reduce_mean(tf.abs(heatmap)))
                attention_variance = float(tf.reduce_mean(tf.square(heatmap - tf.reduce_mean(heatmap))))
                
                return {
                    'attention_score': attention_score,
                    'attention_variance': attention_variance
                }
        
        except Exception as e:
            print(f"Error generating attention maps: {e}")
        
        return {'attention_score': 0.0, 'attention_variance': 0.0}
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def fine_tune(self, unfreeze_layers: int = 20):
        """Unfreeze top layers for fine-tuning."""
        self.base_model.trainable = True
        
        # Freeze bottom layers
        for layer in self.base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
