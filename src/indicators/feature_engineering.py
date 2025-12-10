"""Feature engineering for ML models"""

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
from .technical import TechnicalIndicators


class FeatureEngineer:
    """Feature engineering for machine learning models"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def create_features(self, df: pd.DataFrame, include_time_features: bool = True) -> pd.DataFrame:
        """
        Create comprehensive feature set for ML models
        
        Args:
            df: DataFrame with OHLCV data
            include_time_features: Whether to include time-based features
        
        Returns:
            DataFrame with features added
        """
        df = df.copy()
        
        # Add all technical indicators
        df = self.indicators.add_all_indicators(df)
        
        # Lag features
        df = self._add_lag_features(df)
        
        # Volatility features
        df = self._add_volatility_features(df)
        
        # Time features
        if include_time_features:
            df = self._add_time_features(df)
        
        # Price patterns
        df = self._add_price_patterns(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        # Remove NaN values more selectively
        # Only drop rows where critical features are NaN (OHLCV data)
        critical_cols = ['open', 'high', 'low', 'close', 'volume']
        critical_cols = [col for col in critical_cols if col in df.columns]
        
        # Drop rows where critical columns are NaN
        df = df.dropna(subset=critical_cols)
        
        # For other features, fill NaN with median or forward fill
        for col in df.columns:
            if col not in critical_cols:
                if df[col].dtype in ['float64', 'int64']:
                    # Use forward fill first, then median
                    df[col] = df[col].ffill()
                    df[col] = df[col].fillna(df[col].median())
                    df[col] = df[col].fillna(0)  # Final fallback
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Add lagged price features"""
        for lag in lags:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-related features"""
        # Only add if we have price_change column
        if 'price_change' not in df.columns and 'close' in df.columns:
            df['price_change'] = df['close'].pct_change()
        
        # Rolling volatility - use smaller windows if data is limited
        max_window = min(20, len(df) // 4)
        windows = [w for w in [5, 10, 20] if w <= max_window]
        
        for window in windows:
            if 'price_change' in df.columns:
                df[f'volatility_{window}'] = df['price_change'].rolling(window=window).std()
        
        # High-Low range
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            if len(df) >= 20:
                df['high_low_range_ma'] = df['high_low_range'].rolling(window=20).mean()
            else:
                df['high_low_range_ma'] = df['high_low_range'].rolling(window=min(10, len(df))).mean()
        
        # Price range percentiles
        if all(col in df.columns for col in ['high', 'low', 'close']):
            price_range = df['high'] - df['low']
            df['price_range_percentile'] = (
                (df['close'] - df['low']) / price_range.replace(0, np.nan)
            )
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern features"""
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return df
        
        # Candlestick patterns
        df['body_size'] = abs(df['close'] - df['open']) / df['close'].replace(0, np.nan)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close'].replace(0, np.nan)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close'].replace(0, np.nan)
        
        # Price position in range
        price_range = df['high'] - df['low']
        df['price_position'] = (df['close'] - df['low']) / price_range.replace(0, np.nan)
        
        # Momentum features - use smaller periods if data is limited
        max_period = min(20, len(df) // 4)
        periods = [p for p in [5, 10, 20] if p <= max_period]
        for period in periods:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            shifted_close = df['close'].shift(period)
            df[f'roc_{period}'] = (df['close'] - shifted_close) / shifted_close.replace(0, np.nan) * 100
        
        # Support/Resistance levels (simplified) - use smaller window if data is limited
        window = min(20, len(df) // 4)
        if window >= 5:
            df['resistance'] = df['high'].rolling(window=window).max()
            df['support'] = df['low'].rolling(window=window).min()
            df['distance_to_resistance'] = (df['resistance'] - df['close']) / df['close'].replace(0, np.nan)
            df['distance_to_support'] = (df['close'] - df['support']) / df['close'].replace(0, np.nan)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-related features"""
        if 'volume' not in df.columns:
            return df
        
        # Volume moving averages - use smaller window if data is limited
        window = min(20, len(df) // 4)
        if window >= 5:
            df['volume_sma_20'] = df['volume'].rolling(window=window).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20'].replace(0, np.nan)
        
        # Volume-price relationship
        if 'price_change' in df.columns:
            if window >= 5:
                df['volume_price_trend'] = (df['volume'] * df['price_change']).rolling(window=window).sum()
        
        # On-Balance Volume (OBV)
        if 'close' in df.columns:
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Volume Rate of Change
        max_period = min(20, len(df) // 4)
        periods = [p for p in [5, 10, 20] if p <= max_period]
        for period in periods:
            df[f'volume_roc_{period}'] = df['volume'].pct_change(period)
        
        return df
    
    def create_target_variable(
        self,
        df: pd.DataFrame,
        method: str = 'direction',
        horizon: int = 1
    ) -> pd.Series:
        """
        Create target variable for ML models
        
        Args:
            df: DataFrame with price data
            method: 'direction' (binary: up/down) or 'regression' (continuous price change)
            horizon: Number of periods ahead to predict
        
        Returns:
            Series with target values
        """
        if method == 'direction':
            # Binary classification: 1 if price goes up, 0 if down
            future_price = df['close'].shift(-horizon)
            target = (future_price > df['close']).astype(int)
        elif method == 'regression':
            # Continuous: percentage change
            future_price = df['close'].shift(-horizon)
            target = (future_price - df['close']) / df['close']
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return target
    
    def get_feature_columns(self, df: pd.DataFrame, exclude_target: bool = True) -> List[str]:
        """
        Get list of feature column names
        
        Args:
            df: DataFrame with features
            exclude_target: Whether to exclude target variable columns
        
        Returns:
            List of feature column names
        """
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume',  # Raw OHLCV
            'timestamp', 'target', 'returns', 'log_returns',  # Target/meta columns
            'hour', 'day_of_week', 'day_of_month', 'month'  # Raw time (use encoded versions)
        ]
        
        if exclude_target:
            exclude_cols.extend(['target', 'target_direction', 'target_regression'])
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols

