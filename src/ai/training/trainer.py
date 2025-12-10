"""Model training pipeline"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
import os
from ..models.ml_models import MLModel
from ..models.lstm_model import LSTMModel
from ...indicators.feature_engineering import FeatureEngineer


class ModelTrainer:
    """Trainer for ML models"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize trainer
        
        Args:
            config: Training configuration
        """
        self.config = config or {}
        self.feature_engineer = FeatureEngineer()
        self.models_dir = self.config.get('models_dir', 'models')
        os.makedirs(self.models_dir, exist_ok=True)
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_method: str = 'direction',
        horizon: int = 1
    ) -> tuple:
        """
        Prepare data for training
        
        Args:
            df: DataFrame with OHLCV and indicators
            target_method: 'direction' or 'regression'
            horizon: Prediction horizon
        
        Returns:
            Tuple of (X, y, feature_columns)
        """
        # Create features
        df_features = self.feature_engineer.create_features(df)
        
        # Create target
        target = self.feature_engineer.create_target_variable(
            df_features, method=target_method, horizon=horizon
        )
        
        # Get feature columns
        feature_columns = self.feature_engineer.get_feature_columns(df_features)
        
        # Align data
        df_features = df_features[feature_columns]
        df_features['target'] = target
        
        # Remove NaN and infinity
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        
        # Count rows before dropna
        rows_before = len(df_features)
        
        # Drop rows with NaN, but keep track of what's being dropped
        df_features = df_features.dropna()
        
        rows_after = len(df_features)
        
        if df_features.empty:
            # Provide more diagnostic information
            nan_counts = df_features.isnull().sum()
            inf_counts = (df_features == np.inf).sum() + (df_features == -np.inf).sum()
            raise ValueError(
                f"No valid data after feature engineering. "
                f"Rows before dropna: {rows_before}, after: {rows_after}. "
                f"Original data had {len(df)} rows. "
                f"This usually means there's insufficient data for the feature lookback periods."
            )
        
        X = df_features[feature_columns]
        y = df_features['target']
        
        # Final check for infinity (shouldn't happen after dropna, but just in case)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median()).fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(y.median()).fillna(0)
        
        return X, y, feature_columns
    
    def train_ml_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_config: Optional[Dict] = None,
        model_name: str = 'xgboost'
    ) -> MLModel:
        """
        Train ML model (XGBoost)
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_config: Model configuration
            model_name: Model name for saving
        
        Returns:
            Trained MLModel instance
        """
        model_type = 'classification' if y.dtype == 'int64' or y.dtype == 'bool' else 'regression'
        model = MLModel(config=model_config, model_type=model_type)
        
        # Train
        metrics = model.train(X, y)
        print(f"Training {model_name} - Metrics: {metrics}")
        
        # Save model
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        return model
    
    def train_lstm_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_config: Optional[Dict] = None,
        model_name: str = 'lstm'
    ) -> LSTMModel:
        """
        Train LSTM model
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_config: Model configuration
            model_name: Model name for saving
        
        Returns:
            Trained LSTMModel instance
        """
        model = LSTMModel(config=model_config)
        
        # Train
        metrics = model.train(X, y)
        print(f"Training {model_name} - Metrics: {metrics}")
        
        # Save model
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        return model

