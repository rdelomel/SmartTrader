"""LSTM model for time series prediction"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow - make it optional
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM model will not work. Install with: pip install tensorflow")

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class LSTMModel:
    """LSTM model for sequence prediction"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize LSTM model
        
        Config parameters:
            sequence_length: Number of time steps to look back
            features: Number of features per time step
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate
            dense_units: List of dense layer units
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of training epochs
        """
        self.config = config or {}
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = self.config.get('sequence_length', 60)
        self.features = self.config.get('features', 20)
        self.lstm_units = self.config.get('lstm_units', [64, 32])
        self.dropout_rate = self.config.get('dropout_rate', 0.2)
        self.dense_units = self.config.get('dense_units', [16, 1])
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 32)
        self.epochs = self.config.get('epochs', 50)
        self.is_trained = False
        self.feature_columns = None
    
    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input
        
        Args:
            data: Feature array (n_samples, n_features)
            target: Target array (n_samples,)
        
        Returns:
            Tuple of (X, y) arrays for LSTM
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def _create_model(self, input_shape: Tuple[int, int]):
        """Create LSTM model architecture"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models. Install with: pip install tensorflow")
        
        model = Sequential()
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            if i == 0:
                model.add(LSTM(
                    units,
                    return_sequences=return_sequences,
                    input_shape=input_shape,
                    activation='tanh'
                ))
            else:
                model.add(LSTM(units, return_sequences=return_sequences, activation='tanh'))
            
            model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        for units in self.dense_units[:-1]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(self.dense_units[-1], activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train the LSTM model
        
        Args:
            X: Feature DataFrame
            y: Target Series
            validation_split: Fraction of data for validation
        
        Returns:
            Dictionary with training metrics
        """
        if X.empty or y.empty:
            raise ValueError("Training data cannot be empty")
        
        self.feature_columns = X.columns.tolist()
        
        # Clean data: replace infinity and NaN
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())  # Fill NaN with median
        X_clean = X_clean.fillna(0)  # If still NaN (all values were NaN), fill with 0
        
        y_clean = y.replace([np.inf, -np.inf], np.nan)
        y_clean = y_clean.fillna(y_clean.median())
        y_clean = y_clean.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean.values)
        y_values = y_clean.values.reshape(-1, 1)
        
        # Final check for any remaining infinity or NaN
        if np.any(np.isinf(X_scaled)) or np.any(np.isnan(X_scaled)):
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
        if np.any(np.isinf(y_values)) or np.any(np.isnan(y_values)):
            y_values = np.nan_to_num(y_values, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_values)
        
        if len(X_seq) == 0:
            raise ValueError("Not enough data to create sequences")
        
        # Split data
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Create model
        input_shape = (self.sequence_length, X_scaled.shape[1])
        self.model = self._create_model(input_shape)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            verbose=0,
            shuffle=False
        )
        
        self.is_trained = True
        
        # Calculate final metrics
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        
        return {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': history.history
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.feature_columns is None:
            raise ValueError("Model feature columns not set")
        
        # Ensure same columns as training
        X = X[self.feature_columns].copy()
        
        # Clean data: replace infinity and NaN (same as in training)
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        # Fill NaN with median, then 0 if still NaN
        for col in X_clean.columns:
            if X_clean[col].isnull().any():
                median_val = X_clean[col].median()
                if pd.isna(median_val):
                    X_clean[col] = X_clean[col].fillna(0)
                else:
                    X_clean[col] = X_clean[col].fillna(median_val)
                    X_clean[col] = X_clean[col].fillna(0)  # Final fallback
        
        # Final check: replace any remaining infinity or NaN
        X_values = X_clean.values
        if np.any(np.isinf(X_values)) or np.any(np.isnan(X_values)):
            X_values = np.nan_to_num(X_values, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Scale features
        try:
            X_scaled = self.scaler.transform(X_values)
        except Exception as e:
            # If scaling fails, try with cleaned data again
            X_values = np.nan_to_num(X_values, nan=0.0, posinf=1.0, neginf=-1.0)
            X_scaled = self.scaler.transform(X_values)
        
        # Final safety check after scaling
        if np.any(np.isinf(X_scaled)) or np.any(np.isnan(X_scaled)):
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Create sequences
        if len(X_scaled) < self.sequence_length:
            # Pad with zeros if insufficient data
            padding = np.zeros((self.sequence_length - len(X_scaled), X_scaled.shape[1]))
            X_scaled = np.vstack([padding, X_scaled])
        
        # Get last sequence
        X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        # Predict
        prediction = self.model.predict(X_seq, verbose=0)
        
        return prediction.flatten()
    
    def save(self, filepath: str):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save Keras model using new .keras format (recommended)
        model_path = filepath.replace('.pkl', '_model.keras')
        try:
            # Just use .keras extension - Keras will automatically use the new format
            self.model.save(model_path)
        except Exception as e:
            # Fallback to HDF5 if .keras format fails
            model_path_h5 = filepath.replace('.pkl', '_model.h5')
            try:
                self.model.save(model_path_h5)
                model_path = model_path_h5
            except Exception as e2:
                raise RuntimeError(f"Failed to save model in both .keras and .h5 formats: {e}, {e2}")
        
        # Save scaler and metadata
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'config': self.config,
                'sequence_length': self.sequence_length,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained,
                'model_path': model_path  # Store which format was used
            }, f)
    
    def load(self, filepath: str):
        """Load model from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model metadata file not found: {filepath}")
        
        # Load scaler and metadata first
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.scaler = data['scaler']
            self.config = data['config']
            self.sequence_length = data['sequence_length']
            self.feature_columns = data.get('feature_columns')
            self.is_trained = data.get('is_trained', False)
            model_path = data.get('model_path')
        except Exception as e:
            # If metadata file is corrupted, clean up and retrain
            print(f"Warning: Could not load model metadata ({e}). Will retrain.")
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            self.model = None
            self.is_trained = False
            raise ValueError("Model metadata file is corrupted. Please retrain the model.")
        
        # Determine model path
        if model_path is None:
            # Try new format first, then fallback to old format
            model_path_keras = filepath.replace('.pkl', '_model.keras')
            model_path_h5 = filepath.replace('.pkl', '_model.h5')
            if os.path.exists(model_path_keras):
                model_path = model_path_keras
            elif os.path.exists(model_path_h5):
                model_path = model_path_h5
            else:
                # Model file missing but metadata exists - clean up
                print(f"Warning: Model file not found. Tried: {model_path_keras}, {model_path_h5}. Will retrain.")
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass
                raise FileNotFoundError(f"Model file not found. Tried: {model_path_keras}, {model_path_h5}")
        
        # Load Keras model with error handling for compatibility issues
        try:
            self.model = load_model(model_path, compile=False)
            # Recompile the model to ensure compatibility
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )
        except Exception as e:
            # If loading fails (e.g., due to version incompatibility), delete old model files
            error_msg = str(e)
            print(f"Warning: Could not load saved model ({error_msg}). Will retrain.")
            
            # Clean up all related files
            files_to_remove = [model_path, filepath]
            # Also try to remove alternative format files
            model_path_keras = filepath.replace('.pkl', '_model.keras')
            model_path_h5 = filepath.replace('.pkl', '_model.h5')
            if model_path_keras not in files_to_remove:
                files_to_remove.append(model_path_keras)
            if model_path_h5 not in files_to_remove:
                files_to_remove.append(model_path_h5)
            
            for fpath in files_to_remove:
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                        print(f"Removed incompatible model file: {fpath}")
                    except Exception as rm_error:
                        print(f"Warning: Could not remove {fpath}: {rm_error}")
            
            self.model = None
            self.is_trained = False
            raise ValueError("Model file is incompatible. Please retrain the model.")

