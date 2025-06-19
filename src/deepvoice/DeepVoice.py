import gc
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

import ffmpeg
from dotenv import load_dotenv
from pyannote.audio import Pipeline, Audio, Inference, Model
import torch
from scipy.spatial.distance import cdist
from deepvoice import DiarizationBackend, EmbeddingBackend, VerificationBackend
from deepvoice import PyannoteDiarization, PyannoteEmbedding, PyannoteVerification
from tqdm import tqdm  # Added for progress bars
from . import config

load_dotenv(
    "../../.env",
    override=True,
    verbose=True,
    encoding="utf-8"
)


class DeepVoice:

    def __init__(self,
                 diarization_backend: Optional[DiarizationBackend] = None,
                 embedding_backend: Optional[EmbeddingBackend] = None,
                 verification_backend: Optional[VerificationBackend] = None,
                 hf_token: Optional[str] = None):
        if hf_token is None:
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
                
        if diarization_backend is not None and not isinstance(diarization_backend, DiarizationBackend):
            raise ValueError("diarization_backend must implement DiarizationBackend")
        
        self.diarization = diarization_backend or PyannoteDiarization(hf_token=hf_token)

        if embedding_backend is not None and not isinstance(embedding_backend, EmbeddingBackend):
            raise ValueError("embedding_backend must implement EmbeddingBackend")
        
        self.embedding = embedding_backend or PyannoteEmbedding(hf_token=hf_token)

        if verification_backend is not None and not isinstance(verification_backend, VerificationBackend):
            raise ValueError("verification_backend must implement VerificationBackend")
        
        self.verification = verification_backend or PyannoteVerification(hf_token=hf_token)

    def diarize(self, audio_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Run speaker diarization using the injected backend"""
        return self.diarization.diarize(audio_path, **kwargs)

    def embed(self, audio_path: str, **kwargs) -> Any:
        """Generate voice embedding using the injected backend"""
        return self.embedding.embed(audio_path, **kwargs)

    def verify(self, audio1: Any, audio2: Any, **kwargs) -> float:
        """Compare two audio samples using the injected backend"""
        return self.verification.verify(audio1, audio2, **kwargs)

    @staticmethod
    def extract_voices(
            audio_path: Any,
            model: Optional[str] = config.DEFAULT_DIARIZATION_MODEL,
            hf_token: Optional[str] = None,
            max_speakers: Optional[int] = config.DEFAULT_MAX_SPEAKERS,
            silent: Optional[bool] = config.DEFAULT_SILENT
    ) -> List[Dict[str, Any]] | None:
        if hf_token is None:
            hf_token = os.getenv(config.DEFAULT_HF_TOKEN_ENV_VAR)
        try:
            backend = PyannoteDiarization(hf_token=hf_token, model=model)
            return backend.diarize(audio_path, max_speakers=max_speakers, silent=silent)
        except Exception as e:
            if not silent:
                print(f"Voice processing error: {e}")
            return []
        finally:
            gc.collect()

    @staticmethod
    def represent_voice(
            audio_path: Any,
            embedding_model: str = config.DEFAULT_EMBEDDING_MODEL,
            hf_token: Optional[str] = None,
            silent: bool = config.DEFAULT_SILENT
    ) -> List[Dict[str, Any]] | None:
        if hf_token is None:
            hf_token = os.getenv(config.DEFAULT_HF_TOKEN_ENV_VAR)
        try:
            backend = PyannoteEmbedding(hf_token=hf_token, model=embedding_model)
            embedding = backend.embed(audio_path, silent=silent)
            # Convert numpy array to list for JSON serialization
            return [{"embedding": embedding.tolist()}]
        except Exception as e:
            if not silent:
                print(f"Processing error: {e}")
            return []
        finally:
            gc.collect()

    @staticmethod
    def verify_voice(
            audio1: Any,
            audio2: Any,
            model: str = config.DEFAULT_VERIFICATION_MODEL,
            hf_token: Optional[str] = None,
            silent: bool = config.DEFAULT_SILENT,
            threshold: Optional[float] = config.DEFAULT_THRESHOLD
    ) -> List[Dict[str, Any]] | None:
        if hf_token is None:
            hf_token = os.getenv(config.DEFAULT_HF_TOKEN_ENV_VAR)
        try:
            backend = PyannoteVerification(hf_token=hf_token, model=model)
            distance = backend.verify(audio1, audio2, threshold=threshold, silent=silent)
            # Convert numpy types to native Python types for JSON serialization
            distance = float(distance)
            verified = bool(distance <= threshold)
            return [{"embedding1": audio1, "embedding2": audio2, "distance": distance, "verified": verified}]
        except Exception as e:
            if not silent:
                print(f"Error during verification: {e}")
            return []
        finally:
            gc.collect()

    @staticmethod
    def find_voices(
            audio: Any,
            database_path: Any,
            model: str = config.DEFAULT_VERIFICATION_MODEL,
            hf_token: Optional[str] = None,
            silent: bool = config.DEFAULT_SILENT,
            threshold: Optional[float] = config.DEFAULT_THRESHOLD
    ) -> List[Dict[str, Any]] | None:
        results = []
        if hf_token is None:
            hf_token = os.getenv(config.DEFAULT_HF_TOKEN_ENV_VAR)
        try:
            # Recursively collect all audio files
            audio_exts = config.SUPPORTED_AUDIO_EXTENSIONS
            audio_paths = []
            for root, dirs, files in os.walk(database_path):
                for fname in files:
                    if fname.lower().endswith(audio_exts):
                        audio_paths.append(os.path.join(root, fname))
            # Use ThreadPoolExecutor for parallel verification
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(DeepVoice.verify_voice, audio, path, model, hf_token, silent, threshold): path
                    for path in audio_paths
                }
                for future in concurrent.futures.as_completed(futures):
                    path = futures[future]
                    try:
                        res = future.result()
                        if res:
                            results.extend(res)
                    except Exception as e:
                        if not silent:
                            print(f"Error processing {path}: {e}")
            return results
        except Exception as e:
            if not silent:
                print(f"Processing error: {e}")
            return []
        finally:
            gc.collect()


    @staticmethod
    def represent_emotions(
            audio_path: Any,
            model: Optional[str] = config.DEFAULT_EMOTION_MODEL,
            hf_token: Optional[str] = None,
            silent: Optional[bool] = config.DEFAULT_SILENT
    ) -> List[Dict[str, Any]] | None:

        try:
            if hf_token is None:
                hf_token = os.getenv(config.DEFAULT_HF_TOKEN_ENV_VAR)
            # Configure HuggingFace Hub token if provided
            if hf_token:
                os.environ["HF_HUB_TOKEN"] = hf_token
            # Use SpeechBrain foreign_class for emotion recognition
            from speechbrain.inference.interfaces import foreign_class
            classifier = foreign_class(
                source=model,
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier"
            )
            # Perform emotion classification on file
            out_prob, score, idx, label = classifier.classify_file(audio_path)
            return [{"emotion": label, "confidence": float(score), "path": audio_path}]

        except Exception as e:
            if not silent:
                print(f"Emotion processing error: {str(e)}")
            return []

        finally:
            gc.collect()


    @staticmethod
    def extract_emotions(
            audio_path: Any,
            hf_token: Optional[str] = None,
            max_speakers: Optional[int] = config.DEFAULT_MAX_SPEAKERS,
            silent: Optional[bool] = config.DEFAULT_SILENT
    ) -> List[Dict[str, Any]] | None:

        try:
            # First, extract all voice segments
            voice_segments = DeepVoice.extract_voices(
                audio_path,
                hf_token=hf_token,
                max_speakers=max_speakers,
                silent=silent
            )

            if not voice_segments:
                return []

            # Load emotion model
            from transformers import pipeline
            emotion_classifier = pipeline(
                "audio-classification",
                model="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                token=hf_token
            )

            results = []

            # Analyze each segment with progress bar
            iterator = tqdm(voice_segments, desc="Extracting emotions") if not silent else voice_segments
            for segment in iterator:
                segment_path = segment["path"]
                emotion_result = emotion_classifier(segment_path)

                # Get top emotion
                top_emotion = emotion_result[0]["label"]
                confidence = emotion_result[0]["score"]

                # Add emotion data to segment info
                emotion_segment = segment.copy()
                emotion_segment["emotion"] = top_emotion
                emotion_segment["emotion_confidence"] = confidence

                results.append(emotion_segment)

            return results

        except Exception as e:
            if not silent:
                print(f"Emotional speech analysis error: {str(e)}")
            return []