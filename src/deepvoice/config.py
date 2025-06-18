# Configuration for DeepVoice library

# Environment variable for HuggingFace token
DEFAULT_HF_TOKEN_ENV_VAR = "HUGGINGFACE_TOKEN"

# Supported audio file formats for processing
SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")

# Default model names
DEFAULT_DIARIZATION_MODEL = "speaker-diarization-3.0"
DEFAULT_EMBEDDING_MODEL = "embedding"
DEFAULT_VERIFICATION_MODEL = "embedding"
DEFAULT_EMOTION_MODEL = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

# Default parameters
DEFAULT_MAX_SPEAKERS = 3
DEFAULT_THRESHOLD = 0.5
DEFAULT_SILENT = False 