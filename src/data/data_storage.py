"""Data storage module for historical data and trade records"""

from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Boolean, text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import List, Dict, Optional
import os
import threading
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()


class OHLCVData(Base):
    """OHLCV price data table"""
    __tablename__ = 'ohlcv_data'
    __table_args__ = (
        UniqueConstraint('symbol', 'timeframe', 'timestamp', name='uix_symbol_timeframe_timestamp'),
    )
    
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
    
    @staticmethod
    def _ensure_trade_id_string(trade_id) -> Optional[str]:
        """
        Ensure trade_id is a string (not integer)
        
        Args:
            trade_id: Trade ID (can be string, int, or None)
        
        Returns:
            String trade_id or None
        """
        if trade_id is None:
            return None
        return str(trade_id)

    # Class-level flag and lock to guard lazy table creation from concurrent access
    _tables_created = False
    _tables_lock = threading.Lock()
    
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
            # Mask password in URL for logging
            safe_url = database_url
            if '@' in database_url:
                parts = database_url.split('@')
                if '://' in parts[0]:
                    user_pass = parts[0].split('://')[1]
                    if ':' in user_pass:
                        user = user_pass.split(':')[0]
                        safe_url = database_url.split('://')[0] + '://' + user + ':***@' + '@'.join(parts[1:])
            
            print(f"Connecting to database: {safe_url}")
            
            # For PostgreSQL, use pure-Python pg8000 driver to avoid C-level memory issues
            if database_url.startswith('postgresql://'):
                database_url = database_url.replace('postgresql://', 'postgresql+pg8000://', 1)
                print("Using pg8000 driver for PostgreSQL")
            
            print("Creating database engine...")
            # Choose connect args based on driver (pg8000 does not accept connect_timeout/options)
            connect_args = {}
            if database_url.startswith('postgresql') and '+pg8000' not in database_url:
                connect_args = {
                    "connect_timeout": 10,
                    "options": "-c statement_timeout=30000"  # 30 second statement timeout
                }

            # Use more conservative connection settings to avoid memory issues
            self.engine = create_engine(
                database_url, 
                echo=False, 
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=300,    # Recycle connections after 5 minutes
                pool_size=5,         # Limit pool size
                max_overflow=10,     # Limit overflow
                connect_args=connect_args
            )
            print("Database engine created")
            
            print("Creating session maker...")
            self.Session = sessionmaker(bind=self.engine)
            
            # Create tables immediately to ensure they exist
            print("Creating database tables...")
            self.create_tables()
            
            print("Database initialized successfully")
        except Exception as e:
            print(f"Error initializing database: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def create_tables(self):
        """Explicitly create database tables"""
        if DataStorage._tables_created:
            return

        # Guard against concurrent creation attempts
        with DataStorage._tables_lock:
            if DataStorage._tables_created:
                return

            print("Creating database tables...")
            try:
                # Use engine directly instead of connection to avoid memory issues
                Base.metadata.create_all(self.engine, checkfirst=True)
                DataStorage._tables_created = True
                
                # Verify tables were created by checking if they exist
                from sqlalchemy import inspect
                inspector = inspect(self.engine)
                existing_tables = inspector.get_table_names()
                expected_tables = ['trades', 'ohlcv_data', 'model_predictions']
                
                missing_tables = [t for t in expected_tables if t not in existing_tables]
                if missing_tables:
                    print(f"WARNING: Some tables may not have been created: {missing_tables}")
                    print(f"Existing tables: {existing_tables}")
                else:
                    print(f"Database tables created successfully: {existing_tables}")
                    
            except Exception as table_error:
                # Check if this is actually a connection error
                error_str = str(table_error).lower()
                if any(keyword in error_str for keyword in ['connection', 'connect', 'network', 'timeout', 'refused']):
                    print(f"FATAL: Connection error during table creation: {table_error}")
                    import traceback
                    traceback.print_exc()
                    raise ConnectionError(f"Database connection failed during table creation: {table_error}") from table_error
                
                # For other errors, log but don't fail - tables might already exist
                print(f"Warning: Error creating tables (may already exist): {table_error}")
                import traceback
                traceback.print_exc()
                # Don't mark as created if there was an error - allow retry
                # Only mark as created if we can verify tables exist
                try:
                    from sqlalchemy import inspect
                    inspector = inspect(self.engine)
                    existing_tables = inspector.get_table_names()
                    expected_tables = ['trades', 'ohlcv_data', 'model_predictions']
                    if all(t in existing_tables for t in expected_tables):
                        print(f"Tables verified to exist: {existing_tables}")
                        DataStorage._tables_created = True
                    else:
                        print(f"WARNING: Tables may not exist. Expected: {expected_tables}, Found: {existing_tables}")
                except:
                    pass  # If we can't verify, don't mark as created
    
    def _ensure_tables_created(self):
        """Ensure database tables exist (lazy creation fallback)"""
        if not DataStorage._tables_created:
            self.create_tables()
    
    def get_session(self) -> Session:
        """Get database session"""
        self._ensure_tables_created()  # Ensure tables exist before returning session
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
            
            query = query.order_by(OHLCVData.timestamp.desc())  # Get most recent first
            
            if limit:
                query = query.limit(limit)
            
            # Use timeout protection for long queries
            try:
                records = query.all()
                # Reverse to get chronological order (oldest first)
                records = list(reversed(records))
                return [r.to_dict() for r in records]
            except Exception as e:
                print(f"Error executing query (may be timeout): {e}")
                # Return empty list on error to prevent connection issues
                return []
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
            
            # CRITICAL: Check for existing open position with same (symbol, side, entry_price)
            # This prevents duplicate positions when same order is processed multiple times
            symbol = trade_data.get('symbol')
            side = trade_data.get('side')
            entry_price = trade_data.get('entry_price')
            
            if symbol and side and entry_price:
                existing_open = session.query(Trade).filter(
                    Trade.symbol == symbol,
                    Trade.side == side,
                    Trade.status == 'open',
                    Trade.entry_price >= entry_price * 0.99,
                    Trade.entry_price <= entry_price * 1.01
                ).first()
                
                if existing_open:
                    print(f"  ⚠️  Duplicate open position detected: {symbol} {side} @ ${entry_price:.2f}")
                    print(f"     Updating existing trade {existing_open.id} instead of creating duplicate")
                    
                    # Update existing trade with new SL/TP if provided
                    needs_update = False
                    update_data = {}
                    
                    if trade_data.get('stop_loss') is not None:
                        if existing_open.stop_loss is None or abs(existing_open.stop_loss - trade_data['stop_loss']) > 0.01:
                            update_data['stop_loss'] = trade_data['stop_loss']
                            needs_update = True
                    
                    if trade_data.get('take_profit') is not None:
                        if existing_open.take_profit is None or abs(existing_open.take_profit - trade_data['take_profit']) > 0.01:
                            update_data['take_profit'] = trade_data['take_profit']
                            needs_update = True
                    
                    if needs_update:
                        for key, value in update_data.items():
                            setattr(existing_open, key, value)
                        session.commit()
                        print(f"  ✅ Updated duplicate position {existing_open.id} with SL/TP values: {update_data}")
                    
                    return existing_open.id
            
            # Check if trade already exists by trade_id
            # Ensure trade_id is a string
            trade_id_str = self._ensure_trade_id_string(trade_id) or f"trade_{datetime.now().timestamp()}"
            existing_trade = session.query(Trade).filter_by(trade_id=trade_id_str).first()
            if existing_trade:
                # Trade already exists - update if stop_loss/take_profit are missing
                needs_update = False
                update_data = {}
                
                # Update stop_loss if provided and currently missing
                if trade_data.get('stop_loss') is not None and existing_trade.stop_loss is None:
                    update_data['stop_loss'] = trade_data['stop_loss']
                    needs_update = True
                
                # Update take_profit if provided and currently missing
                if trade_data.get('take_profit') is not None and existing_trade.take_profit is None:
                    update_data['take_profit'] = trade_data['take_profit']
                    needs_update = True
                
                # Also update if new values are different (in case they were set incorrectly)
                if trade_data.get('stop_loss') is not None and existing_trade.stop_loss is not None:
                    if abs(existing_trade.stop_loss - trade_data['stop_loss']) > 0.01:
                        update_data['stop_loss'] = trade_data['stop_loss']
                        needs_update = True
                
                if trade_data.get('take_profit') is not None and existing_trade.take_profit is not None:
                    if abs(existing_trade.take_profit - trade_data['take_profit']) > 0.01:
                        update_data['take_profit'] = trade_data['take_profit']
                        needs_update = True
                
                if needs_update:
                    for key, value in update_data.items():
                        setattr(existing_trade, key, value)
                    session.commit()
                    print(f"  ✅ Updated existing trade {existing_trade.id} (trade_id: {trade_id_str}) with SL/TP values: {update_data}")
                
                return existing_trade.id
            
            trade = Trade(
                trade_id=trade_id_str,
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
            trade_id: Trade ID to update (will be converted to string)
            update_data: Dictionary with fields to update
        
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session()
        
        try:
            # Ensure trade_id is a string (not integer)
            trade_id_str = self._ensure_trade_id_string(trade_id)
            if not trade_id_str:
                return False
            
            trade = session.query(Trade).filter_by(trade_id=trade_id_str).first()
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
            trade_id: Trade ID to delete (will be converted to string)
        
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session()
        
        try:
            # Ensure trade_id is a string (not integer)
            trade_id_str = self._ensure_trade_id_string(trade_id)
            if not trade_id_str:
                return False
            
            trade = session.query(Trade).filter_by(trade_id=trade_id_str).first()
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