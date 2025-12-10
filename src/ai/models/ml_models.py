"""Machine learning models (XGBoost, scikit-learn)"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import pickle
import os
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class MLModel:
    """XGBoost-based ML model for price direction prediction"""
    
    def __init__(self, config: Optional[Dict] = None, model_type: str = 'classification'):
        """
        Initialize ML model
        
        Args:
            config: Model configuration dictionary
            model_type: 'classification' (up/down) or 'regression' (price change)
        """
        self.config = config or {}
        self.model_type = model_type
        self.model = None
        self.feature_importance_ = None
        self.is_trained = False
        
        # XGBoost parameters
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', 6)
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.subsample = self.config.get('subsample', 0.8)
        self.colsample_bytree = self.config.get('colsample_bytree', 0.8)
        self.min_child_weight = self.config.get('min_child_weight', 3)
    
    def _create_model(self):
        """Create XGBoost model instance"""
        if self.model_type == 'classification':
            self.model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                min_child_weight=self.min_child_weight,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1
            )
        else:  # regression
            self.model = XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                min_child_weight=self.min_child_weight,
                objective='reg:squarederror',
                eval_metric='rmse',
                random_state=42,
                n_jobs=-1
            )
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        early_stopping_rounds: int = 10
    ) -> Dict:
        """
        Train the model
        
        Args:
            X: Feature DataFrame
            y: Target Series
            validation_split: Fraction of data to use for validation
            early_stopping_rounds: Early stopping rounds
        
        Returns:
            Dictionary with training metrics
        """
        if self.model is None:
            self._create_model()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, shuffle=False
        )
        
        # Train model
        if self.model_type == 'classification':
            # XGBoost 2.0+ uses callbacks for early stopping
            # Try multiple approaches for compatibility
            try:
                from xgboost import callback
                callbacks = [callback.EarlyStopping(rounds=early_stopping_rounds)]
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=callbacks,
                    verbose=False
                )
            except (ImportError, AttributeError, TypeError):
                # Fallback 1: Try early_stopping_rounds parameter
                try:
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                    )
                except TypeError:
                    # Fallback 2: Train without early stopping
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
            
            # Calculate metrics
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            self.feature_importance_ = pd.Series(
                self.model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            
            self.is_trained = True
            
            return {
                'accuracy': accuracy,
                'feature_importance': self.feature_importance_.to_dict()
            }
        else:  # regression
            # Try multiple approaches for compatibility
            try:
                from xgboost import callback
                callbacks = [callback.EarlyStopping(rounds=early_stopping_rounds)]
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=callbacks,
                    verbose=False
                )
            except (ImportError, AttributeError, TypeError):
                # Fallback 1: Try early_stopping_rounds parameter
                try:
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                    )
                except TypeError:
                    # Fallback 2: Train without early stopping
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
            
            # Calculate metrics
            y_pred = self.model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            self.feature_importance_ = pd.Series(
                self.model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            
            self.is_trained = True
            
            return {
                'rmse': rmse,
                'feature_importance': self.feature_importance_.to_dict()
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
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities (for classification)
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of probability predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model_type != 'classification':
            raise ValueError("predict_proba only available for classification models")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.Series:
        """
        Get top feature importances
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            Series with feature importances
        """
        if self.feature_importance_ is None:
            return pd.Series()
        
        return self.feature_importance_.head(top_n)
    
    def save(self, filepath: str):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'config': self.config,
                'feature_importance': self.feature_importance_,
                'is_trained': self.is_trained
            }, f)
    
    def load(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.model_type = data['model_type']
            self.config = data['config']
            self.feature_importance_ = data.get('feature_importance')
            self.is_trained = data.get('is_trained', False)

