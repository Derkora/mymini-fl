from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from .db import Base
from datetime import datetime

class Client(Base):
    __tablename__ = "clients"

    id = Column(String, primary_key=True)
    ip_address = Column(String)
    last_seen = Column(DateTime, default=datetime.utcnow)
    fl_status = Column(String)

    trainings = relationship("TrainingUpdate", back_populates="client")


class UserGlobal(Base):
    __tablename__ = "users_global"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String)
    identity = Column(String, unique=True)

    attendance = relationship("AttendanceRecap", back_populates="user")


class AttendanceRecap(Base):
    __tablename__ = "attendance_recap"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users_global.id"))
    client_id = Column(String, ForeignKey("clients.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("UserGlobal", back_populates="attendance")


class TrainingRound(Base):
    __tablename__ = "training_rounds"

    id = Column(Integer, primary_key=True)
    round_number = Column(Integer)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    model_version = Column(String)


class TrainingUpdate(Base):
    __tablename__ = "training_updates"

    id = Column(Integer, primary_key=True)
    round_id = Column(Integer, ForeignKey("training_rounds.id"))
    client_id = Column(String, ForeignKey("clients.id"))
    update_metadata = Column(JSON)

    client = relationship("Client", back_populates="trainings")
