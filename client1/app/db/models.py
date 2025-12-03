from sqlalchemy import Column, Integer, String, DateTime, LargeBinary, Float
from sqlalchemy.orm import relationship
from .db import Base
from datetime import datetime

class UserLocal(Base):
    __tablename__ = "users_local"

    id = Column(Integer, primary_key=True)
    identity = Column(String, unique=True)
    name = Column(String)

    embeddings = relationship("Embedding", back_populates="user")
    attendance = relationship("AttendanceLocal", back_populates="user")


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    vector = Column(LargeBinary)

    user = relationship("UserLocal", back_populates="embeddings")


class AttendanceLocal(Base):
    __tablename__ = "attendance_local"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("UserLocal", back_populates="attendance")


class LocalConfig(Base):
    __tablename__ = "local_config"

    id = Column(Integer, primary_key=True)
    last_training = Column(DateTime)
    total_samples = Column(Integer)
