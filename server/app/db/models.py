from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Float, LargeBinary, Text
from sqlalchemy.orm import relationship
from .db import Base
from datetime import datetime

class Client(Base):
    __tablename__ = "clients"
    # SRS 1.1: Tabel Clients
    edge_id = Column(String, primary_key=True) # Mengganti 'id' menjadi 'edge_id' sesuai SRS
    name = Column(String)
    ip_address = Column(String)
    status = Column(String) # online/offline
    last_seen = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    attendance_recap = relationship("AttendanceRecap", back_populates="client")
    training_updates = relationship("TrainingUpdate", back_populates="client")

class UserGlobal(Base):
    __tablename__ = "users_global"
    # SRS 1.2: Tabel Users Global
    user_id = Column(Integer, primary_key=True, autoincrement=True) # Mengganti 'id'
    name = Column(String)
    nim = Column(String) # Tambahan NIM
    registered_edge_id = Column(String, ForeignKey("clients.edge_id"))
    created_at = Column(DateTime, default=datetime.utcnow)

    attendance = relationship("AttendanceRecap", back_populates="user")

class AttendanceRecap(Base):
    __tablename__ = "attendance_recap"
    # SRS 1.3: Tabel Attendance Recap
    recap_id = Column(Integer, primary_key=True, index=True) # Mengganti 'id'
    user_id = Column(Integer, ForeignKey("users_global.user_id"))
    edge_id = Column(String, ForeignKey("clients.edge_id")) # Mengganti client_id
    timestamp = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float) # Tambahan sesuai SRS
    lecture_id = Column(String, nullable=True)

    user = relationship("UserGlobal", back_populates="attendance")
    client = relationship("Client", back_populates="attendance_recap")

class ModelVersion(Base):
    __tablename__ = "model_versions"
    # SRS 1.4: Tabel Model Versions (PENTING untuk FL)
    version_id = Column(Integer, primary_key=True, autoincrement=True)
    head_blob = Column(LargeBinary) # Menyimpan bobot model (pickle/bytes)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)

    rounds = relationship("TrainingRound", back_populates="model_version")

class TrainingRound(Base):
    __tablename__ = "training_rounds"
    # SRS 1.5: Tabel Training Rounds
    round_id = Column(Integer, primary_key=True) # Mengganti id
    round_number = Column(Integer)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    global_loss = Column(Float)
    global_accuracy = Column(Float)
    
    # Relasi ke versi model yang dihasilkan ronde ini
    model_version_id = Column(Integer, ForeignKey("model_versions.version_id"))
    
    model_version = relationship("ModelVersion", back_populates="rounds")
    updates = relationship("TrainingUpdate", back_populates="round")

class TrainingUpdate(Base):
    __tablename__ = "training_updates"
    # SRS 1.6: Tabel Training Updates (Log kiriman client)
    update_id = Column(Integer, primary_key=True)
    round_id = Column(Integer, ForeignKey("training_rounds.round_id"))
    edge_id = Column(String, ForeignKey("clients.edge_id"))
    
    payload_size = Column(Integer) # Ukuran byte
    upload_time = Column(Float) # Detik
    loss_local = Column(Float)
    accuracy_local = Column(Float)

    client = relationship("Client", back_populates="training_updates")
    round = relationship("TrainingRound", back_populates="updates")