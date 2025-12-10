"""Anomaly detection using Isolation Forest"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """Detect anomalies in market data"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize anomaly detector
        
        Config parameters:
            contamination: Expected proportion of anomalies (default: 0.1)
            n_estimators: Number of trees (default: 100)
            max_samples: Maximum samples per tree (default: 256)
        """
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self.contamination = self.config.get('contamination', 0.1)
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_samples = self.config.get('max_samples', 256)
    
    def train(self, data: pd.DataFrame, features: Optional[List[str]] = None):
        """
        Train anomaly detection model
        
        Args:
            data: DataFrame with market data
            features: List of feature columns to use (None for auto-selection)
        """
        if features is None:
            # Default features for anomaly detection
            features = ['price_change', 'volume_change', 'volatility']
            # Only use features that exist
            features = [f for f in features if f in data.columns]
        
        if not features:
            raise ValueError("No valid features found for anomaly detection")
        
        # Extract features
        X = data[features].copy()
        X = X.fillna(0)  # Fill NaN with 0
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled)
        self.is_trained = True
    
    def detect(self, data: pd.DataFrame, features: Optional[List[str]] = None) -> pd.Series:
        """
        Detect anomalies in data
        
        Args:
            data: DataFrame with market data
            features: List of feature columns to use
        
        Returns:
            Boolean Series indicating anomalies (True = anomaly)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before detection")
        
        if features is None:
            features = ['price_change', 'volume_change', 'volatility']
            features = [f for f in features if f in data.columns]
        
        if not features:
            return pd.Series([False] * len(data), index=data.index)
        
        # Extract features
        X = data[features].copy()
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict anomalies
        predictions = self.model.predict(X_scaled)
        
        # Convert to boolean (1 = normal, -1 = anomaly)
        anomalies = predictions == -1
        
        return pd.Series(anomalies, index=data.index)
    
    def get_anomaly_score(self, data: pd.DataFrame, features: Optional[List[str]] = None) -> pd.Series:
        """
        Get anomaly scores (lower = more anomalous)
        
        Args:
            data: DataFrame with market data
            features: List of feature columns to use
        
        Returns:
            Series with anomaly scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting scores")
        
        if features is None:
            features = ['price_change', 'volume_change', 'volatility']
            features = [f for f in features if f in data.columns]
        
        if not features:
            return pd.Series([0.0] * len(data), index=data.index)
        
        # Extract features
        X = data[features].copy()
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores
        scores = self.model.score_samples(X_scaled)
        
        return pd.Series(scores, index=data.index)
    
    def is_anomaly(self, data_point: Dict) -> bool:
        """
        Check if a single data point is an anomaly
        
        Args:
            data_point: Dictionary with feature values
        
        Returns:
            True if anomaly, False otherwise
        """
        if not self.is_trained:
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame([data_point])
        
        # Detect
        anomalies = self.detect(df)
        
        return anomalies.iloc[0] if not anomalies.empty else False

