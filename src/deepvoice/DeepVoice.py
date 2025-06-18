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
            model: Optional[str] = "speaker-diarization-3.0",
            hf_token: Optional[str] = None,
            max_speakers: Optional[int] = 3,
            silent: Optional[bool] = False
    ) -> List[Dict[str, Any]] | None:
        if hf_token is None:
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
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
            embedding_model: str="embedding",
            hf_token: Optional[str] = None,
            silent: bool = False
    ) -> List[Dict[str, Any]] | None:
        if hf_token is None:
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
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
            model: str = "embedding",
            hf_token: Optional[str] = None,
            silent: bool = False,
            threshold: Optional[float] = 0.5
    ) -> List[Dict[str, Any]] | None:
        if hf_token is None:
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
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
            model: str = "embedding",
            hf_token: Optional[str] = None,
            silent: bool = False,
            threshold: Optional[float] = 0.5
    ) -> List[Dict[str, Any]] | None:
        results = []

        try:

            if hf_token is None:
                hf_token = os.getenv("HUGGINGFACE_TOKEN")

            embed_model = Model.from_pretrained(
                f"pyannote/{model}",
                use_auth_token=hf_token,
                strict=False
            )
            inference = Inference(embed_model, window="whole")

            audio1_path = None
            audio2_path = None

            try:

                if isinstance(audio, str):
                    audio1_path = audio
                    embedding1 = inference(audio)
                    embedding1 = embedding1.reshape(1, -1)
                else:
                    embedding1 = audio.reshape(1, -1)

                # Prepare list of .wav files
                wav_files = [f for f in os.listdir(database_path) if f.endswith(".wav")]
                iterator = tqdm(wav_files, desc="Finding voices") if not silent else wav_files
                for file in iterator:
                    embedding2 = inference(os.path.join(database_path, file))
                    embedding2 = embedding2.reshape(1, -1)
                    # Compute and convert numpy types to native Python types for JSON serialization
                    raw_distance = cdist(embedding1, embedding2, metric="cosine")[0, 0]
                    distance = float(raw_distance)
                    verified = bool(distance <= threshold)
                    audio2_path = os.path.join(database_path, file)
                    results.append({
                        "embedding1": audio1_path,
                        "embedding2": audio2_path,
                        "distance": distance,
                        "verified": verified
                    })

                return results

            except Exception as e:
                if not silent:
                    print(f"Error during verification: {str(e)}")
                    import traceback
                    traceback.print_exc()  # Show full traceback
                return []

        except Exception as e:
            if not silent:
                print(f"Processing error: {str(e)}")
            return []

        finally:
            gc.collect()


    @staticmethod
    def represent_emotions(
            audio_path: Any,
            model: Optional[str] = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            hf_token: Optional[str] = None,
            silent: Optional[bool] = False
    ) -> List[Dict[str, Any]] | None:

        try:
            if hf_token is None:
                hf_token = os.getenv("HUGGINGFACE_TOKEN")

            # Load a pre-trained emotion recognition model
            from transformers import pipeline
            classifier = pipeline(
                "audio-classification",
                model=model,
                token=hf_token
            )

            # Perform emotion classification
            emotion_result = classifier(audio_path)

            # Format results
            results = [{
                "emotion": item["label"],
                "confidence": item["score"],
                "path": audio_path
            } for item in emotion_result]

            return results

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
            max_speakers: Optional[int] = 3,
            silent: Optional[bool] = False
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