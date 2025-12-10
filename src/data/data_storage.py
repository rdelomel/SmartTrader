"""Data storage module for historical data and trade records"""

from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()


class OHLCVData(Base):
    """OHLCV price data table"""
    __tablename__ = 'ohlcv_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    timeframe = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


class Trade(Base):
    """Trade records table"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String, unique=True, nullable=False, index=True)
    symbol = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False)  # 'buy' or 'sell'
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    entry_time = Column(DateTime, nullable=False, index=True)
    exit_time = Column(DateTime, nullable=True)
    status = Column(String, nullable=False)  # 'open', 'closed', 'stopped', 'taken_profit'
    pnl = Column(Float, default=0.0)
    pnl_percent = Column(Float, default=0.0)
    strategy = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelPrediction(Base):
    """Model predictions table"""
    __tablename__ = 'model_predictions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    model_name = Column(String, nullable=False)
    prediction = Column(Float, nullable=False)  # Predicted price change or direction
    confidence = Column(Float, nullable=True)
    actual_price = Column(Float, nullable=True)  # Actual price for validation
    created_at = Column(DateTime, default=datetime.utcnow)


class DataStorage:
    """Data storage manager for trading data"""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize data storage
        
        Args:
            database_url: SQLAlchemy database URL. Defaults to SQLite if not provided.
        """
        if database_url is None:
            database_url = os.getenv('DATABASE_URL', 'sqlite:///./data/trading.db')
        
        # Create data directory if it doesn't exist
        if database_url.startswith('sqlite'):
            os.makedirs('data', exist_ok=True)
            # Extract database file path for logging
            db_path = database_url.replace('sqlite:///', '')
            if not os.path.exists(db_path):
                print(f"Creating SQLite database at: {db_path}")
            else:
                print(f"Using existing SQLite database at: {db_path}")
        
        try:
            self.engine = create_engine(database_url, echo=False)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            print("Database initialized successfully")
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.Session()
    
    def store_ohlcv_data(self, symbol: str, timeframe: str, data: List[Dict]) -> int:
        """
        Store OHLCV data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            data: List of OHLCV dictionaries
        
        Returns:
            Number of records stored
        """
        session = self.get_session()
        count = 0
        
        try:
            for record in data:
                # Check if record already exists
                existing = session.query(OHLCVData).filter_by(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=record['timestamp']
                ).first()
                
                if not existing:
                    ohlcv = OHLCVData(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=record['timestamp'],
                        open=record['open'],
                        high=record['high'],
                        low=record['low'],
                        close=record['close'],
                        volume=record['volume']
                    )
                    session.add(ohlcv)
                    count += 1
            
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error storing OHLCV data: {e}")
        finally:
            session.close()
        
        return count
    
    def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve OHLCV data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of records
        
        Returns:
            List of OHLCV dictionaries
        """
        session = self.get_session()
        
        try:
            query = session.query(OHLCVData).filter_by(
                symbol=symbol,
                timeframe=timeframe
            )
            
            if start_date:
                query = query.filter(OHLCVData.timestamp >= start_date)
            if end_date:
                query = query.filter(OHLCVData.timestamp <= end_date)
            
            query = query.order_by(OHLCVData.timestamp.asc())
            
            if limit:
                query = query.limit(limit)
            
            records = query.all()
            return [r.to_dict() for r in records]
        except Exception as e:
            print(f"Error retrieving OHLCV data: {e}")
            return []
        finally:
            session.close()
    
    def store_trade(self, trade_data: Dict) -> Optional[int]:
        """
        Store trade record
        
        Args:
            trade_data: Dictionary with trade information
        
        Returns:
            Trade ID if successful, None otherwise
        """
        session = self.get_session()
        
        try:
            trade_id = trade_data.get('trade_id', f"trade_{datetime.now().timestamp()}")
            
            # Check if trade already exists
            existing_trade = session.query(Trade).filter_by(trade_id=trade_id).first()
            if existing_trade:
                # Trade already exists, return existing ID
                return existing_trade.id
            
            trade = Trade(
                trade_id=trade_id,
                symbol=trade_data['symbol'],
                side=trade_data['side'],
                quantity=trade_data['quantity'],
                entry_price=trade_data['entry_price'],
                exit_price=trade_data.get('exit_price'),
                stop_loss=trade_data.get('stop_loss'),
                take_profit=trade_data.get('take_profit'),
                entry_time=trade_data.get('entry_time', datetime.now()),
                exit_time=trade_data.get('exit_time'),
                status=trade_data.get('status', 'open'),
                pnl=trade_data.get('pnl', 0.0),
                pnl_percent=trade_data.get('pnl_percent', 0.0),
                strategy=trade_data.get('strategy')
            )
            session.add(trade)
            session.commit()
            return trade.id
        except Exception as e:
            session.rollback()
            print(f"Error storing trade: {e}")
            return None
        finally:
            session.close()
    
    def update_trade(self, trade_id: str, update_data: Dict) -> bool:
        """
        Update trade record
        
        Args:
            trade_id: Trade ID to update
            update_data: Dictionary with fields to update
        
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session()
        
        try:
            trade = session.query(Trade).filter_by(trade_id=trade_id).first()
            if trade:
                for key, value in update_data.items():
                    if hasattr(trade, key):
                        setattr(trade, key, value)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Error updating trade: {e}")
            return False
        finally:
            session.close()
    
    def delete_trade(self, trade_id: str) -> bool:
        """
        Delete a trade record (for phantom positions that don't exist in broker)
        
        Args:
            trade_id: Trade ID to delete
        
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session()
        
        try:
            trade = session.query(Trade).filter_by(trade_id=trade_id).first()
            if trade:
                session.delete(trade)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Error deleting trade: {e}")
            return False
        finally:
            session.close()
    
    def get_open_trades(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get all open trades
        
        Args:
            symbol: Optional symbol filter
        
        Returns:
            List of trade dictionaries
        """
        session = self.get_session()
        
        try:
            query = session.query(Trade).filter_by(status='open')
            if symbol:
                query = query.filter_by(symbol=symbol)
            
            trades = query.all()
            return [{
                'id': t.id,
                'trade_id': t.trade_id,
                'symbol': t.symbol,
                'side': t.side,
                'quantity': t.quantity,
                'entry_price': t.entry_price,
                'stop_loss': t.stop_loss,
                'take_profit': t.take_profit,
                'entry_time': t.entry_time.isoformat() if t.entry_time else None,
                'pnl': t.pnl,
                'strategy': t.strategy
            } for t in trades]
        except Exception as e:
            print(f"Error retrieving open trades: {e}")
            return []
        finally:
            session.close()
    
    def get_all_trades(self, symbol: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all trades (both open and closed)
        
        Args:
            symbol: Optional symbol filter
            limit: Optional limit on number of trades to return
        
        Returns:
            List of trade dictionaries
        """
        session = self.get_session()
        
        try:
            query = session.query(Trade)
            if symbol:
                query = query.filter_by(symbol=symbol)
            
            # Order by entry_time descending (most recent first)
            query = query.order_by(Trade.entry_time.desc())
            
            if limit:
                query = query.limit(limit)
            
            trades = query.all()
            return [{
                'id': t.id,
                'trade_id': t.trade_id,
                'symbol': t.symbol,
                'side': t.side,
                'quantity': t.quantity,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'stop_loss': t.stop_loss,
                'take_profit': t.take_profit,
                'entry_time': t.entry_time.isoformat() if t.entry_time else None,
                'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                'status': t.status,
                'pnl': t.pnl,
                'strategy': t.strategy
            } for t in trades]
        except Exception as e:
            print(f"Error retrieving all trades: {e}")
            return []
        finally:
            session.close()
    
    def store_prediction(self, prediction_data: Dict) -> Optional[int]:
        """
        Store model prediction
        
        Args:
            prediction_data: Dictionary with prediction information
        
        Returns:
            Prediction ID if successful, None otherwise
        """
        session = self.get_session()
        
        try:
            prediction = ModelPrediction(
                symbol=prediction_data['symbol'],
                timestamp=prediction_data.get('timestamp', datetime.now()),
                model_name=prediction_data['model_name'],
                prediction=prediction_data['prediction'],
                confidence=prediction_data.get('confidence'),
                actual_price=prediction_data.get('actual_price')
            )
            session.add(prediction)
            session.commit()
            return prediction.id
        except Exception as e:
            session.rollback()
            print(f"Error storing prediction: {e}")
            return None
        finally:
            session.close()

