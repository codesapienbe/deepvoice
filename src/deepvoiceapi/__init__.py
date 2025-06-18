from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from celery import chain
from celery.result import AsyncResult
import time
import uvicorn
import tempfile
from pathlib import Path
import logging
import logging.config
import zipfile
import os

from deepvoiceworker import convert_to_wav_task, extract_voices_task, represent_voice_task, verify_voice_task, find_voices_task, represent_emotions_task, extract_emotions_task, task_service

# Ensure log directory exists
LOG_DIR = Path.home() / ".deepvoice" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
        }
    },
    "handlers": {
        "core_file": {
            "class": "logging.FileHandler",
            "filename": str(LOG_DIR / "core.log"),
            "formatter": "standard",
            "level": "DEBUG"
        },
        "api_file": {
            "class": "logging.FileHandler",
            "filename": str(LOG_DIR / "api.log"),
            "formatter": "standard",
            "level": "INFO"
        },
        "tasks_file": {
            "class": "logging.FileHandler",
            "filename": str(LOG_DIR / "tasks.log"),
            "formatter": "standard",
            "level": "INFO"
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "DEBUG",
            "stream": "ext://sys.stdout"
        }
    },
    "loggers": {
        "deepvoice": {
            "handlers": ["core_file", "console"],
            "level": "DEBUG",
            "propagate": False
        },
        "uvicorn": {
            "handlers": ["api_file", "console"],
            "level": "INFO",
            "propagate": False
        },
        "celery": {
            "handlers": ["tasks_file", "console"],
            "level": "INFO",
            "propagate": False
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)

# Initialize FastAPI app
api_service = FastAPI(title="DeepVoice API")

# Add CORS middleware
api_service.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Added request models for new DeepVoice functions
class ExtractVoicesRequest(BaseModel):
    audio_path: str
    model: Optional[str] = None
    hf_token: Optional[str] = None
    max_speakers: Optional[int] = None
    silent: Optional[bool] = None

class RepresentVoiceRequest(BaseModel):
    audio_path: str
    embedding_model: Optional[str] = None
    hf_token: Optional[str] = None
    silent: Optional[bool] = None

class VerifyVoiceRequest(BaseModel):
    audio1: str
    audio2: str
    model: Optional[str] = None
    hf_token: Optional[str] = None
    silent: Optional[bool] = None
    threshold: Optional[float] = None

class FindVoicesRequest(BaseModel):
    audio: str
    database_path: str
    model: Optional[str] = None
    hf_token: Optional[str] = None
    silent: Optional[bool] = None
    threshold: Optional[float] = None

class RepresentEmotionsRequest(BaseModel):
    audio_path: str
    model: Optional[str] = None
    hf_token: Optional[str] = None
    silent: Optional[bool] = None

class ExtractEmotionsRequest(BaseModel):
    audio_path: str
    hf_token: Optional[str] = None
    max_speakers: Optional[int] = None
    silent: Optional[bool] = None

# API endpoints with file upload support and automatic conversion
@api_service.post("/extract_voices")
async def extract_voices_api(
    file: UploadFile = File(...),
    model: str = Form("speaker-diarization-3.0"),
    hf_token: Optional[str] = Form(None),
    max_speakers: int = Form(3),
    silent: bool = Form(False),
):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    contents = await file.read()
    tmp.write(contents)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()
    params = {
        "audio_path": tmp_path,
        "model": model,
        "hf_token": hf_token,
        "max_speakers": max_speakers,
        "silent": silent,
    }
    # Chain conversion then processing
    result = chain(
        convert_to_wav_task.s(params),
        extract_voices_task.s()
    )
    async_result = result.delay()
    return {"task_id": async_result.id}

@api_service.post("/represent_voice")
async def represent_voice_api(
    file: UploadFile = File(...),
    embedding_model: str = Form("embedding"),
    hf_token: Optional[str] = Form(None),
    silent: bool = Form(False),
):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    contents = await file.read()
    tmp.write(contents)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()
    params = {
        "audio_path": tmp_path,
        "embedding_model": embedding_model,
        "hf_token": hf_token,
        "silent": silent,
    }
    # Chain conversion then processing
    async_result = chain(
        convert_to_wav_task.s(params),
        represent_voice_task.s()
    ).delay()
    return {"task_id": async_result.id}

@api_service.post("/verify_voice")
async def verify_voice_api(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    model: str = Form("embedding"),
    hf_token: Optional[str] = Form(None),
    silent: bool = Form(False),
    threshold: float = Form(0.5),
):
    tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file1.filename).suffix)
    tmp1.write(await file1.read())
    tmp1.flush()
    path1 = tmp1.name
    tmp1.close()
    tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file2.filename).suffix)
    tmp2.write(await file2.read())
    tmp2.flush()
    path2 = tmp2.name
    tmp2.close()
    params = {
        "audio1": path1,
        "audio2": path2,
        "model": model,
        "hf_token": hf_token,
        "silent": silent,
        "threshold": threshold,
    }
    # Chain conversion then processing
    async_result = chain(
        convert_to_wav_task.s(params),
        verify_voice_task.s()
    ).delay()
    return {"task_id": async_result.id}

@api_service.post("/find_voices")
async def find_voices_api(
    file: UploadFile = File(...),
    zip_file: UploadFile = File(...),
    model: str = Form("embedding"),
    hf_token: Optional[str] = Form(None),
    silent: bool = Form(False),
    threshold: float = Form(0.5),
):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    tmp.write(await file.read())
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()
    # Save and extract uploaded zip as database
    zip_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    zip_tmp.write(await zip_file.read())
    zip_tmp.flush()
    zip_path = zip_tmp.name
    zip_tmp.close()
    # Create temporary directory for database audio
    db_temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(db_temp_dir)
    # Remove the zip file
    os.remove(zip_path)
    # Delete non-audio files from the database directory
    for root, dirs, files in os.walk(db_temp_dir):
        for fname in files:
            if not fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")):
                os.remove(os.path.join(root, fname))
    params = {
        "audio": tmp_path,
        "database_path": db_temp_dir,
        "model": model,
        "hf_token": hf_token,
        "silent": silent,
        "threshold": threshold,
    }
    # Chain conversion then processing
    async_result = chain(
        convert_to_wav_task.s(params),
        find_voices_task.s()
    ).delay()
    return {"task_id": async_result.id}

@api_service.post("/represent_emotions")
async def represent_emotions_api(
    file: UploadFile = File(...),
    model: str = Form("speechbrain/emotion-recognition-wav2vec2-IEMOCAP"),
    hf_token: Optional[str] = Form(None),
    silent: bool = Form(False),
):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    tmp.write(await file.read())
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()
    params = {
        "audio_path": tmp_path,
        "model": model,
        "hf_token": hf_token,
        "silent": silent,
    }
    # Chain conversion then processing
    async_result = chain(
        convert_to_wav_task.s(params),
        represent_emotions_task.s()
    ).delay()
    return {"task_id": async_result.id}

@api_service.post("/extract_emotions")
async def extract_emotions_api(
    file: UploadFile = File(...),
    hf_token: Optional[str] = Form(None),
    max_speakers: int = Form(3),
    silent: bool = Form(False),
):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    tmp.write(await file.read())
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()
    params = {
        "audio_path": tmp_path,
        "hf_token": hf_token,
        "max_speakers": max_speakers,
        "silent": silent,
    }
    # Chain conversion then processing
    async_result = chain(
        convert_to_wav_task.s(params),
        extract_emotions_task.s()
    ).delay()
    return {"task_id": async_result.id}

@api_service.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=task_service)
    if task_result.ready():
        if task_result.successful():
            return {"status": "completed", "result": task_result.result}
        return {"status": "failed", "error": str(task_result.result)}
    return {"status": "pending"}

# Health check endpoint
@api_service.get("/health")
async def health_check():
    return {"status": "healthy"}

def log_start_time(start_time: float):
    print(f"Starting DeepVoice API at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

def log_end_time(start_time: float):
    end_time = time.time()
    print(f"DeepVoice API exited at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    active_time = end_time - start_time
    days, remainder = divmod(active_time, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"DeepVoice API was active for {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds")

def main():
    start_time = time.time()
    try:
        log_start_time(start_time)
        uvicorn.run(api_service, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Error starting DeepVoice API: {e}")
    finally:
        log_end_time(start_time)

if __name__ == "__main__":
    main()