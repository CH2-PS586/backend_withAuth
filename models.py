from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

from database import Base


class Users(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    username = Column(String(300),unique=True)
    hashed_password = Column(String(255))

#     File = relationship("Item", back_populates="owner")


class File(Base):
    __tablename__ = "files"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255))
    file_size = Column(Integer)
    category = Column(String(255), nullable=False)
    label = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    gcp_bucket_url = Column(String(255))

    # owner_id = Column(Integer, ForeignKey("users.id"))
    # owner = relationship("User", back_populates="files")