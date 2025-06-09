#database_module/db.py
import os
from sqlalchemy import create_engine, inspect, text
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

def apply_migrations():
    """
    Apply any necessary migrations to existing tables.
    """
    with engine.connect() as conn:
        # Check if the models table exists and has the capabilities column
        inspector = inspect(engine)
        if "models" in inspector.get_table_names():
            columns = [col['name'] for col in inspector.get_columns('models')]
            if "capabilities" not in columns:
                # Add capabilities column to models table
                conn.execute(text("ALTER TABLE models ADD COLUMN capabilities TEXT"))
                conn.commit()
                print("Migration: Added capabilities column to models table")

def init_db():
    """
    Create tables if they don't exist.
    Call this once at application startup.
    """
    Base.metadata.create_all(bind=engine)
    apply_migrations()
