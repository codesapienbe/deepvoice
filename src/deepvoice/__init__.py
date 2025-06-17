from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
import os
import gc
from datetime import datetime
import ffmpeg
from pyannote.audio import Pipeline, Audio, Model, Inference

# Base Backend Classes
class DiarizationBackend(ABC):
    @abstractmethod
    def diarize(self, audio_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Segment an audio file by speaker."""
        pass

class EmbeddingBackend(ABC):
    @abstractmethod
    def embed(self, audio_path: str, **kwargs) -> np.ndarray:
        """Generate an embedding vector for an audio file."""
        pass

class VerificationBackend(ABC):
    @abstractmethod
    def verify(self, audio1: str, audio2: str, **kwargs) -> float:
        """Compare two audio files and return a distance metric."""
        pass

# Pyannote Backend Implementations
class PyannoteDiarization(DiarizationBackend):
    def __init__(self, hf_token: str, model: str = "speaker-diarization-3.0"):
        self.hf_token = hf_token
        self.model = model
        self.pipeline = Pipeline.from_pretrained(
            f"pyannote/{model}", use_auth_token=hf_token
        )
        self.audio = Audio()

    def diarize(self, audio_path: str, max_speakers: int = 3, silent: bool = False, **kwargs) -> List[Dict[str, Any]]:
        try:
            duration = self.audio.get_duration(audio_path)
            diarization = self.pipeline(
                file=audio_path,
                min_speakers=1,
                max_speakers=max_speakers
            )
            results: List[Dict[str, Any]] = []
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = os.path.expanduser("~/.deepvoice")
            voices_dir = os.path.join(base_dir, "voices")
            os.makedirs(voices_dir, exist_ok=True)

            for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
                start = min(turn.start, duration)
                end = min(turn.end, duration)
                if end <= start + 0.01:
                    continue
                filename = f"{session_id}_{speaker}_{i:04d}_{start:.3f}_{end:.3f}.wav"
                voice_path = os.path.join(voices_dir, filename)
                ffmpeg.input(audio_path).output(
                    voice_path, ss=start, to=end
                ).overwrite_output().run(quiet=True)
                results.append({
                    "speaker": speaker,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "path": voice_path.replace("\\", "/"),
                })
            return results
        except Exception as e:
            if not silent:
                print(f"Voice processing error: {e}")
            return []
        finally:
            gc.collect()

class PyannoteEmbedding(EmbeddingBackend):
    def __init__(self, hf_token: str, model: str = "embedding"):
        self.hf_token = hf_token
        self.model = model
        self.embed_model = Model.from_pretrained(
            f"pyannote/{model}", use_auth_token=hf_token, strict=False
        )
        self.inference = Inference(self.embed_model, window="whole")

    def embed(self, audio_path: str, silent: bool = False, **kwargs) -> np.ndarray:
        try:
            embedding = self.inference(audio_path)
            return embedding
        except Exception as e:
            if not silent:
                print(f"Embedding processing error: {e}")
            return np.array([])

class PyannoteVerification(VerificationBackend):
    def __init__(self, hf_token: str, model: str = "embedding"):
        self.hf_token = hf_token
        self.model = model
        self.embed_backend = PyannoteEmbedding(hf_token=hf_token, model=model)

    def verify(self, audio1: str, audio2: str, threshold: float = 0.5, silent: bool = False, **kwargs) -> float:
        try:
            emb1 = self.embed_backend.embed(audio1, silent=silent)
            emb2 = self.embed_backend.embed(audio2, silent=silent)
            emb1 = emb1.reshape(1, -1)
            emb2 = emb2.reshape(1, -1)
            from scipy.spatial.distance import cdist
            distance = cdist(emb1, emb2, metric="cosine")[0, 0]
            return distance
        except Exception as e:
            if not silent:
                print(f"Verification processing error: {e}")
            return float('inf')

# Export DeepVoice class
from .DeepVoice import DeepVoice

__all__ = ['DeepVoice', 'DiarizationBackend', 'EmbeddingBackend', 'VerificationBackend',
           'PyannoteDiarization', 'PyannoteEmbedding', 'PyannoteVerification']

__version__ = '0.1.0'

