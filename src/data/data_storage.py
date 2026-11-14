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
