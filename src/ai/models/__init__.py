"""AI models module"""

from .ml_models import MLModel
try:
    from .lstm_model import LSTMModel
except ImportError:
    LSTMModel = None  # LSTM model optional if TensorFlow not available
from .sentiment_analyzer import SentimentAnalyzer

__all__ = ['MLModel', 'LSTMModel', 'SentimentAnalyzer']

