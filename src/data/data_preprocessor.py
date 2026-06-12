"""Data preprocessing and cleaning module"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime


class DataPreprocessor:
    """Data preprocessing and cleaning utilities"""
    
    @staticmethod
    def clean_ohlcv_data(data: List[Dict]) -> pd.DataFrame:
        """
        Clean and convert OHLCV data to DataFrame
        
        Args:
            data: List of OHLCV dictionaries
        
        Returns:
            Cleaned pandas DataFrame
        """
        # Safe empty-check: handles both DataFrames and lists
        if data is None:
            return pd.DataFrame()
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return pd.DataFrame()
        elif not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Remove rows with invalid data
        df = df[
            (df['open'] > 0) &
            (df['high'] > 0) &
            (df['low'] > 0) &
            (df['close'] > 0) &
            (df['volume'] >= 0) &
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        ]
        
        # Forward fill missing values (limited)
        df.ffill(limit=3, inplace=True)
        
        # Drop any remaining NaN values
        df.dropna(inplace=True)
        
        return df
    
    @staticmethod
    def normalize_data(df: pd.DataFrame, method: str = 'min_max') -> pd.DataFrame:
        """
        Normalize price data
        
        Args:
            df: DataFrame with price data
            method: Normalization method ('min_max', 'z_score', 'robust')
        
        Returns:
            Normalized DataFrame
        """
        df_normalized = df.copy()
        
        price_columns = ['open', 'high', 'low', 'close']
        
        if method == 'min_max':
            for col in price_columns:
                if col in df_normalized.columns:
                    min_val = df_normalized[col].min()
                    max_val = df_normalized[col].max()
                    if max_val > min_val:
                        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        elif method == 'z_score':
            for col in price_columns:
                if col in df_normalized.columns:
                    mean = df_normalized[col].mean()
                    std = df_normalized[col].std()
                    if std > 0:
                        df_normalized[col] = (df_normalized[col] - mean) / std
        
        elif method == 'robust':
            for col in price_columns:
                if col in df_normalized.columns:
                    median = df_normalized[col].median()
                    iqr = df_normalized[col].quantile(0.75) - df_normalized[col].quantile(0.25)
                    if iqr > 0:
                        df_normalized[col] = (df_normalized[col] - median) / iqr
        
        return df_normalized
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, method: str = 'iqr') -> pd.Series:
        """
        Detect outliers in price data
        
        Args:
            df: DataFrame with price data
            method: Outlier detection method ('iqr', 'z_score')
        
        Returns:
            Boolean Series indicating outliers
        """
        if method == 'iqr':
            Q1 = df['close'].quantile(0.25)
            Q3 = df['close'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (df['close'] < lower_bound) | (df['close'] > upper_bound)
        
        elif method == 'z_score':
            z_scores = np.abs((df['close'] - df['close'].mean()) / df['close'].std())
            return z_scores > 3
        
        return pd.Series([False] * len(df), index=df.index)
    
    @staticmethod
    def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample data to different timeframe
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Target timeframe (e.g., '1h', '4h', '1d')
        
        Returns:
            Resampled DataFrame
        """
        if df.empty:
            return df
        
        # Convert timeframe to pandas offset
        timeframe_map = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1w': '1W'
        }
        
        offset = timeframe_map.get(timeframe, '1H')
        
        resampled = df.resample(offset).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Remove rows with NaN (incomplete periods)
        resampled.dropna(inplace=True)
        
        return resampled
    
    @staticmethod
    def add_returns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add return columns to DataFrame
        
        Args:
            df: DataFrame with price data
        
        Returns:
            DataFrame with return columns added
        """
        df = df.copy()
        
        # Simple returns
        df['returns'] = df['close'].pct_change()
        
        # Log returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Cumulative returns
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        return df
    
    @staticmethod
    def add_price_changes(df: pd.DataFrame, periods: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Add price change features for different periods
        
        Args:
            df: DataFrame with price data
            periods: List of periods to calculate changes
        
        Returns:
            DataFrame with price change columns added
        """
        df = df.copy()
        
        for period in periods:
            df[f'price_change_{period}'] = df['close'].pct_change(period)
            df[f'price_change_abs_{period}'] = abs(df[f'price_change_{period}'])
        
        return df
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate data quality
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'has_data': len(df) > 0,
            'no_nulls': not df.isnull().any().any(),
            'positive_prices': (df[['open', 'high', 'low', 'close']] > 0).all().all(),
            'valid_ohlc': (
                (df['high'] >= df['low']) &
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close'])
            ).all(),
            'has_volume': (df['volume'] >= 0).all(),
            'no_duplicates': not df.index.duplicated().any()
        }
        
        results['is_valid'] = all(results.values())
        
        return results

