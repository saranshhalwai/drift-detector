import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configure your database URL (env var or fallback to SQLite file)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./drift_detector.sqlite3"
)

# Create engine and session factory
# For SQLite, disable same-thread check
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)
Base = declarative_base()

def init_db():
    """
    Create tables if they don't exist.
    Call this once at application startup.
    """
    Base.metadata.create_all(bind=engine)
