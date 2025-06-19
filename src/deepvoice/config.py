# Configuration for DeepVoice library

# Environment variable for HuggingFace token
DEFAULT_HF_TOKEN_ENV_VAR = "HUGGINGFACE_TOKEN"

# Supported audio file formats for processing
SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm", ".m4b")

# Default model names
DEFAULT_DIARIZATION_MODEL = "speaker-diarization-3.0"
DEFAULT_EMBEDDING_MODEL = "embedding"
DEFAULT_VERIFICATION_MODEL = "embedding"
DEFAULT_EMOTION_MODEL = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

# Default parameters
DEFAULT_MAX_SPEAKERS = 3
DEFAULT_THRESHOLD = 0.5
DEFAULT_SILENT = False

# Add import for environment variables
import os

# Redis and Celery settings
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "0")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
if REDIS_PASSWORD:
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
else:
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL) 