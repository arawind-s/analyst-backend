from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, JSON, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:user@localhost/analyst")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# User Model
class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user")
    datasets = relationship("Dataset", back_populates="user")

# Conversation Model
class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

# Message Model
class Message(Base):
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    chart_data = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

# Dataset Model
class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    row_count = Column(Integer)
    columns = Column(JSON)
    dtypes = Column(JSON)
    upload_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default='active')

    # Relationships
    user = relationship("User", back_populates="datasets")

# SavedChart Model
class SavedChart(Base):
    __tablename__ = "saved_charts"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=True)
    message_id = Column(String, ForeignKey("messages.id"), nullable=True)
    title = Column(String, nullable=False)
    chart_type = Column(String, nullable=False)  # 'bar', 'line', 'pie', etc.
    chart_data = Column(JSON, nullable=False)
    interpretation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_favorite = Column(Integer, default=0)  # 0 = not favorite, 1 = favorite

    # Relationships
    user = relationship("User")
    conversation = relationship("Conversation")
    message = relationship("Message")

# Dashboard Model - Saved dashboards with charts and configurations
class Dashboard(Base):
    __tablename__ = "dashboards"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    title = Column(String, default="New Dashboard")
    charts = Column(JSON, nullable=False)  # List of chart objects with positions
    background_pattern = Column(String, default="transparent")  # Background pattern/color
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User")

def init_db():
    """Initialize database and run migrations"""
    Base.metadata.create_all(bind=engine)

    # Migration: Add background_pattern column to dashboards table if it doesn't exist
    with engine.connect() as conn:
        try:
            # Check if column exists
            check_query = text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name='dashboards' AND column_name='background_pattern'
            """)
            result = conn.execute(check_query)

            if result.fetchone() is None:
                # Column doesn't exist, add it
                print("Adding background_pattern column to dashboards table...")
                alter_query = text("""
                    ALTER TABLE dashboards
                    ADD COLUMN background_pattern VARCHAR DEFAULT 'transparent'
                """)
                conn.execute(alter_query)
                conn.commit()
                print("✓ Successfully added background_pattern column to dashboards table!")
        except Exception as e:
            print(f"Migration note: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()