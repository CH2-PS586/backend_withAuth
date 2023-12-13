from typing import Optional
from pydantic import BaseModel
from datetime import datetime

class FileBase(BaseModel):
    filename: str
    file_size: int
    category: str
    label: Optional[str] = None
    gcp_bucket_url: str

class FileCreate(FileBase):
    pass

class FileUpdate(FileBase):
    pass

class FileInDB(FileBase):
    id: int
    created_at: datetime
    updated_at: datetime
    gcp_bucket_url: str

class FileOut(FileInDB):
    owner_id: Optional[int]

    class Config:
        orm_mode = True