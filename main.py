from fastapi import FastAPI, UploadFile, File, Depends
from fastapi import HTTPException, Path
from fastapi.responses import JSONResponse
from google.cloud import storage
import os
#from dotenv import load_dotenv
import json
from typing import Annotated, Optional
from fastapi import Query
# Library and dependency for Machine Learning
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Database Stuff
from database import SessionLocal, engine
import models
from models import File as FileModel
from schemas import FileCreate
from sqlalchemy.orm import Session
from fastapi.encoders import jsonable_encoder

from starlette.responses import StreamingResponse
import io

# Authorization Stuff
import auth
from auth import get_current_user
from starlette import status

models.Base.metadata.create_all(bind=engine)

# Fill with your project id 
os.environ["GOOGLE_CLOUD_PROJECT"] = ""

def get_db():
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()

label_array = ['Collage', 'Food', 'Friends', 'Memes', 'Pets', 'Selfie']

class HubLayer(tf.keras.layers.Layer):
    def __init__(self, handle, **kwargs):
        self.handle = handle
        super(HubLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.hub_layer = hub.KerasLayer(self.handle, trainable=False)
        super(HubLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.hub_layer(inputs)

    def get_config(self):
        config = super(HubLayer, self).get_config()
        config.update({"handle": self.handle})
        return config

# Replace 'path/to/your/model.h5' with the actual path to your HDF5 model file
model_path = 'model_file/modeltype1.h5'

# Define a custom object scope to tell TensorFlow about the custom layer
custom_objects = {'HubLayer': HubLayer}

# Load the model using the custom object scope
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model(model_path)

def prepare_data_for_prediction(img_content):
    img_array = img_to_array(img_content)
    img_array = tf.image.resize(img_array, (128, 128))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(np.copy(img_array))
    return img_array

# Load environment variables from .env file
#load_dotenv()

app = FastAPI()

app.include_router(auth.router)

db_dependency = Annotated[Session, Depends(get_db)] 
user_dependency = Annotated[dict, Depends(get_current_user)]

# Fetch sensitive information from environment variables
#key_json_content = os.getenv('KEY_CONFIG')
#bucket_name = os.getenv('BUCKET_NAME')

client = storage.Client.from_service_account_json("")
bucket_name = ''

# Convert the environment variable containing JSON to a Python dictionary
#key_info = json.loads(key_json_content)

# Use the fetched values
#KEY = {}
#client = storage.Client.from_service_account_info(KEY)

def get_category(file_extension: str):
    music_extensions = ['mp3', 'wav', 'ogg']
    picture_extensions = ['jpg', 'jpeg', 'png', 'gif']
    document_extensions = ['pdf', 'doc', 'docx', 'txt']
    video_extensions = ['mp4', 'avi', 'mkv']

    if file_extension.lower() in music_extensions:
        return 'music'
    elif file_extension.lower() in picture_extensions:
        return 'picture'
    elif file_extension.lower() in document_extensions:
        return 'document'
    elif file_extension.lower() in video_extensions:
        return 'video'
    else:
        return 'others'

def upload_file_to_gcs(file: UploadFile, category: str, label: str, username: str):

    if label is not None:
        blob_name = f"{username}/{category}/{label}/{file.filename}"
    else:
        blob_name = f"{username}/{category}/{file.filename}"

    blob = client.bucket(bucket_name).blob(blob_name)

    # Ensure the file stream is at the beginning
    file.file.seek(0)

    blob.upload_from_file(file.file)

    gcs_url = f"gs://{bucket_name}/{blob_name}"
    return {"message": f"File uploaded successfully to {category} category and {label} label" if label else f"File uploaded successfully to {category} category", "gcs_url": gcs_url}

def save_file_to_database(db: Session, username: str, filename: str, file_size: int, category: str, label: str = None, gcs_url: str = None):
    file_data = FileCreate(
        username=username,
        filename=filename,
        file_size=file_size,
        category=category,
        label=label,
        gcp_bucket_url=gcs_url
    )
    db_file = FileModel(**file_data.dict())  # Replace 1 with the actual user ID
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return db_file

def convert_bytes_to_human_readable(size_in_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0

def create_user_folder(username, bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    folder_name = f"{username}/"

    if not any(blob.name.startswith(folder_name) for blob in bucket.list_blobs()):
        # The folder doesn't exist, so create it
        blob = bucket.blob(folder_name)
        blob.upload_from_string("")
        print(f"Folder created: {folder_name}")

@app.post("/files")
async def upload(user: dict = Depends(get_current_user), files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    if user is None:
        raise HTTPException(status_code=401, detail='Forbidden. You Need to Login First')

    create_user_folder(user['username'], bucket_name)

    messages = []
    for file in files:
        file_extension = file.filename.split('.')[-1]
        category = get_category(file_extension)
        label = None
        if category == 'picture':
            file_content = await file.read()
            img_content = tf.image.decode_image(file_content, channels=3)
            prepare_image = prepare_data_for_prediction(img_content)
            prediction = model.predict(prepare_image)
            label = str(label_array[np.argmax(prediction)])

        # Upload the file to GCS
        upload_result = upload_file_to_gcs(file, category, label, user["username"])
        gcs_url = upload_result.get("gcs_url", "")

        # Save file information to the database including the GCS URL
        db_file = save_file_to_database(db, user['username'], file.filename, file.file._file.tell(), category, label, gcs_url)
        
        messages.append({"upload_message": upload_result["message"],
                         "category": category,
                         "label": label,
                         "db_record": jsonable_encoder(db_file)})

    return messages

ALLOWED_CATEGORIES = {"music", "video", "others"}
ML_CATEGORIES = {"picture", "document"}

ALLOWED_LABELS = ['Collage', 'Food', 'Friends', 'Memes', 'Pets', 'Selfie', 'Personal', 'Work or School']

@app.get("/files/{category}")
async def get_files_by_category(user: dict = Depends(get_current_user), category: str = Path(...,title="Category"),
    db: Session = Depends(get_db)
):
    try:
        # Check if the specified category is allowed
        if category.lower() not in ALLOWED_CATEGORIES:
            raise HTTPException(status_code=400, detail="Invalid category")

        # If the category is "picture," return the list of allowed labels
        #if category.lower() == "picture":
            #return {"allowed_labels": ALLOWED_LABELS}

        # List files in the specified category from the cloud storage bucket
        blobs = client.bucket(bucket_name).list_blobs(prefix=f"{user['username']}/{category.lower()}/")


        # Extract file information
        files_info = []
        for blob in blobs:
            files_info.append({
                "filename": blob.name.split("/")[-1],
                "file_size": convert_bytes_to_human_readable(blob.size),
                "gcs_url": f"gs://{bucket_name}/{blob.name}",
                "created_at": blob.time_created,
                "updated_at": blob.updated,
            })

        return files_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{category}")
async def get_files_by_category_test(user: dict = Depends(get_current_user), category: str = Path(...,title="Category"), db: Session = Depends(get_db)):
    try:
        # Check if the specified category is allowed
        if category.lower() not in ALLOWED_CATEGORIES:
            raise HTTPException(status_code=400, detail="Invalid category")

        # List files in the specified category from the cloud storage bucket, including subfolders
        prefix = f"{user['username']}/{category.lower()}/"
        blobs = client.bucket(bucket_name).list_blobs(prefix=prefix, delimiter='/')

        # Extract file information
        files_info = []
        for blob in blobs:
            if blob.name.endswith('/'):
                # This is a subfolder, you can choose to handle it differently if needed
                continue

            files_info.append({
                "filename": blob.name.split("/")[-1],
                "file_size": convert_bytes_to_human_readable(blob.size),
                "gcs_url": f"gs://{bucket_name}/{blob.name}",
                "created_at": blob.time_created,
                "updated_at": blob.updated,
            })

        return files_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{category}/{label}")
async def get_files_by_category_and_label(user: dict = Depends(get_current_user), category: str = Path(..., title="Category"), label: str = Path(..., title="Label")):
    try:
        # Check if the specified category is allowed
        if category.lower() not in ML_CATEGORIES:
            raise HTTPException(status_code=400, detail="Invalid category")
        
        # Check if the specified label is allowed
        if label not in ALLOWED_LABELS:
            raise HTTPException(status_code=400, detail="Invalid label")        

        # List files in the specified category from the cloud storage bucket
        blobs = client.bucket(bucket_name).list_blobs(prefix=f"{user['username']}/{category.lower()}/{label}/")


        # Extract file information
        files_info = []
        for blob in blobs:
            files_info.append({
                "filename": blob.name.split("/")[-1],
                "file_size": convert_bytes_to_human_readable(blob.size),
                "gcs_url": f"gs://{bucket_name}/{blob.name}",
                "created_at": blob.time_created,
                "updated_at": blob.updated,
            })

        return files_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/getfiles/{category}/{filename}")
async def get_file_content(user: dict = Depends(get_current_user), category: str = Path(..., title="Category"), filename: str = Path(..., title="Filename")):
    try:
        # Check if the specified category is allowed
        if category.lower() not in ALLOWED_CATEGORIES:
            raise HTTPException(status_code=400, detail="Invalid category")

        # Check if the file exists in the specified category
        blob_name = f"{user['username']}/{category.lower()}/{filename}"
        blob = client.bucket(bucket_name).blob(blob_name)
        if not blob.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Retrieve the content of the file as bytes
        file_content = blob.download_as_bytes()

        # Create a StreamingResponse to send binary data
        return StreamingResponse(io.BytesIO(file_content), media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="{filename}"'})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/getfiles/{category}/{label}/{filename}")
async def get_file_content(user: dict = Depends(get_current_user), category: str = Path(..., title="Category"), label: str = Path(..., title="Label"), filename: str = Path(..., title="Filename")):
    try:
        # Check if the specified category is allowed
        if category.lower() not in ML_CATEGORIES:
            raise HTTPException(status_code=400, detail="Invalid category")

        # Check if the specified label is allowed
        if label not in ALLOWED_LABELS:
            raise HTTPException(status_code=400, detail="Invalid label")

        # Check if the file exists in the specified label within the "picture" category
        blob_name = f"{user['username']}/{category.lower()}/{label}/{filename}"
        blob = client.bucket(bucket_name).blob(blob_name)
        if not blob.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Retrieve the content of the file as bytes
        file_content = blob.download_as_bytes()

        # Create a StreamingResponse to send binary data
        return StreamingResponse(io.BytesIO(file_content), media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="{filename}"'})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", status_code=status.HTTP_200_OK)
async def verify_status(user:user_dependency, db: db_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail='Forbidden. You Need to Login First')
    return {"User": user}