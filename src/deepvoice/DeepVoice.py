import gc
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

import ffmpeg
from dotenv import load_dotenv
from pyannote.audio import Pipeline, Audio, Inference, Model
import torch
from scipy.spatial.distance import cdist

load_dotenv(
    "../../.env",
    override=True,
    verbose=True,
    encoding="utf-8"
)


class DeepVoice:

    @staticmethod
    def extract_voices(
            audio_path: Any,
            model: Optional[str] = "speaker-diarization-3.0",
            hf_token: Optional[str] = None,
            max_speakers: Optional[int] = 3,
            silent: Optional[bool] = False
    ) -> List[Dict[str, Any]] | None:

        try:
            if hf_token is None:
                hf_token = os.getenv("HUGGINGFACE_TOKEN")

            # Create the base directory and voices subdirectory if they don't exist
            base_dir = os.path.expanduser("~/.deepvoice")
            voices_dir = os.path.join(base_dir, "voices")
            os.makedirs(voices_dir, exist_ok=True)

            diarization_pipeline = Pipeline.from_pretrained(
                f"pyannote/{model}",
                use_auth_token=hf_token
            )

            audio = Audio()
            duration = audio.get_duration(audio_path)

            diarization = diarization_pipeline(
                file=audio_path,
                min_speakers=1,
                max_speakers=max_speakers,
            )

            results = []

            # Generate a unique session ID to group related voice segments
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
                start = min(turn.start, duration)
                end = min(turn.end, duration)
                if end <= start + 0.01:  # 10ms minimum duration
                    continue

                # Create a filename with session, speaker and timing information
                filename = f"{session_id}_{speaker}_{i:04d}_{start:.3f}_{end:.3f}.wav"
                voice_path = os.path.join(voices_dir, filename)

                # Use ffmpeg for audio trimming
                (
                    ffmpeg
                    .input(audio_path)
                    .output(voice_path, ss=start, to=end)
                    .overwrite_output()
                    .run(quiet=True)
                )

                segment_details = {
                    "speaker": speaker,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "path": voice_path.replace("\\", "/"),
                }
                results.append(segment_details)

            return results

        except Exception as e:
            if not silent:
                print(f"Voice processing error: {str(e)}")
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
        """
        Extract speaker embeddings using pyannote's embedding model.
        # 1. visit hf.co/pyannote/embedding and accept user conditions
        # 2. visit hf.co/settings/tokens to create an access token
        # 3. instantiate pretrained model
        """
        results = []
        try:

            if hf_token is None:
                hf_token = os.getenv("HUGGINGFACE_TOKEN")

            embed_model = Model.from_pretrained(
                f"pyannote/{embedding_model}",
                use_auth_token=hf_token,
                strict=False
            )
            inference = Inference(embed_model, window="whole")
            embedding = inference(audio_path)
            results.append({
                "embedding": embedding
            })
            return results

        except Exception as e:
            if not silent:
                print(f"Processing error: {str(e)}")
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
        """
        Extract speaker embeddings' using pyannote's embedding model, then compare them.
        # 1. visit hf.co/pyannote/embedding and accept user conditions
        # 2. visit hf.co/settings/tokens to create an access token
        # 3. instantiate pretrained model
        """
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

                if isinstance(audio1, str):
                    embedding1 = inference(audio1)
                    embedding1 = embedding1.reshape(1, -1)
                    audio1_path = audio1
                else:
                    embedding1 = audio1.reshape(1, -1)

                if isinstance(audio2, str):
                    embedding2 = inference(audio2)
                    embedding2 = embedding2.reshape(1, -1)
                    audio2_path = audio2
                else:
                    embedding2 = audio2.reshape(1, -1)

                distance = cdist(embedding1, embedding2, metric="cosine")[0, 0]

                # NOTE: ≤ 0.5: Often considered a “same speaker” indicator.

                results.append({
                    "embedding1": audio1_path,
                    "embedding2": audio2_path,
                    "distance": distance,
                    "verified": distance <= threshold
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

                for file in os.listdir(database_path):
                    if file.endswith(".wav"):
                        embedding2 = inference(os.path.join(database_path, file))
                        embedding2 = embedding2.reshape(1, -1)
                        distance = cdist(embedding1, embedding2, metric="cosine")[0, 0]
                        audio2_path = os.path.join(database_path, file)
                        results.append({
                            "embedding1": audio1_path,
                            "embedding2": audio2_path,
                            "distance": distance,
                            "verified": distance <= threshold
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

            # Analyze each segment
            for segment in voice_segments:
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