from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from celery import Celery, chain
from celery.result import AsyncResult
import time
import uvicorn
from deepvoice import DeepVoice
import tempfile
from pathlib import Path
import ffmpeg  # for audio conversion
import threading  # for embedded Celery worker

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

# Initialize Celery
task_service = Celery(
    'deepvoice',
    broker='memory://',
    backend='rpc://'
)

# Configure Celery
task_service.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Initialize DeepVoice
deepvoice = DeepVoice()

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

# Celery task: convert various audio formats to WAV before processing
@task_service.task
def convert_to_wav_task(params: Dict[str, Any]) -> Dict[str, Any]:
    # Check and convert any audio keys in params
    for key in ('audio_path', 'audio', 'audio1', 'audio2'):
        path = params.get(key)
        if isinstance(path, str) and not path.lower().endswith('.wav'):
            output_path = f"{path}.wav"
            # Convert to WAV
            ffmpeg.input(path).output(output_path, format='wav').run(overwrite_output=True)
            params[key] = output_path
    return params

# Added Celery tasks for new DeepVoice functions
@task_service.task
def extract_voices_task(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = deepvoice.extract_voices(**params)
        return {"status": "success", "result": result}
    except Exception as e:
        raise

@task_service.task
def represent_voice_task(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = deepvoice.represent_voice(**params)
        return {"status": "success", "result": result}
    except Exception as e:
        raise

@task_service.task
def verify_voice_task(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = deepvoice.verify_voice(**params)
        return {"status": "success", "result": result}
    except Exception as e:
        raise

@task_service.task
def find_voices_task(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = deepvoice.find_voices(**params)
        return {"status": "success", "result": result}
    except Exception as e:
        raise

@task_service.task
def represent_emotions_task(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = deepvoice.represent_emotions(**params)
        return {"status": "success", "result": result}
    except Exception as e:
        raise

@task_service.task
def extract_emotions_task(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = deepvoice.extract_emotions(**params)
        return {"status": "success", "result": result}
    except Exception as e:
        raise

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
    database_path: str = Form(...),
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
    params = {
        "audio": tmp_path,
        "database_path": database_path,
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

# Function to start a Celery worker in-process
def start_celery_worker():
    # Run the Celery worker with solo pool to avoid forking
    task_service.worker_main(argv=['worker', '--loglevel=info', '--pool=solo'])

def main():
    try:
        # Start time
        start_time = time.time()
        print(f"Starting DeepVoice API and embedded Celery worker at {start_time}")
        # Launch Celery worker in background thread
        worker_thread = threading.Thread(
            target=start_celery_worker,
            daemon=True
        )
        worker_thread.start()
        # Launch FastAPI server (blocks until shutdown)
        uvicorn.run(api_service, host="0.0.0.0", port=8000)
        end_time = time.time()
        print(f"DeepVoice API exited at {end_time}")
        active_time = end_time - start_time
        days, remainder = divmod(active_time, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"DeepVoice API was active for {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds")
    except Exception as e:
        print(f"Error starting DeepVoice API: {e}")

if __name__ == "__main__":
    main()