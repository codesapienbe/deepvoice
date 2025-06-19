from celery import Celery
from typing import Dict, Any
import ffmpeg  # for audio conversion
from deepvoice import DeepVoice
from pytz import timezone
from deepvoice.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

# Initialize Celery using environment-configured URLs
task_service = Celery(
    'deepvoice',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

# Configure Celery
task_service.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone=timezone('Europe/Brussels'),
    enable_utc=True,
)

# Initialize DeepVoice
deepvoice = DeepVoice()

@task_service.task
def convert_to_wav_task(params: Dict[str, Any]) -> Dict[str, Any]:
    # Check and convert any audio keys in params
    for key in ('audio_path', 'audio', 'audio1', 'audio2'):
        path = params.get(key)
        if isinstance(path, str) and not path.lower().endswith('.wav'):
            output_path = f"{path}.wav"
            ffmpeg.input(path).output(output_path, format='wav').run(overwrite_output=True)
            params[key] = output_path
    return params

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

def start_celery_worker():
    # run Celery worker in-process with solo pool
    task_service.worker_main(argv=['worker', '--loglevel=info', '--pool=solo'])

def main():
    start_celery_worker()

if __name__ == "__main__":
    main()
