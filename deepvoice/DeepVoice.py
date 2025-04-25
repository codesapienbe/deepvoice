import os
import tempfile
from datetime import datetime
from typing import Optional, List, Dict, Any

import ffmpeg
from pyannote.audio import Pipeline, Audio, Inference, Model
from pyannote.core import Segment
import numpy as np

import gc

from dotenv import load_dotenv

load_dotenv(
    "../.env",
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
    def represent(
            audio_path: Any,
            embedding_model: str="embedding",
            hf_token: Optional[str] = None,
            silent: bool = False,
            gpu: Optional[bool] = False
    ) -> List[Dict[str, Any]] | None:
        """
        Extract speaker embeddings using pyannote's embedding model.
        # 1. visit hf.co/pyannote/embedding and accept user conditions
        # 2. visit hf.co/settings/tokens to create an access token
        # 3. instantiate pretrained model
        """
        results = []
        try:

            if gpu:
                import torch
                torch.cuda.empty_cache()

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
    def verify(
            audio1_path: Any,
            audio2_path: Any,
            embedding_model: str = "embedding",
            hf_token: Optional[str] = None,
            silent: bool = False,
            threshold: Optional[float] = 0.5,
            gpu: Optional[bool] = False
    ) -> List[Dict[str, Any]] | None:
        """
        Extract speakers embeddings' using pyannote's embedding model, then compare them.
        # 1. visit hf.co/pyannote/embedding and accept user conditions
        # 2. visit hf.co/settings/tokens to create an access token
        # 3. instantiate pretrained model
        """
        results = []
        import torch
        from scipy.spatial.distance import cdist

        try:

            if hf_token is None:
                hf_token = os.getenv("HUGGINGFACE_TOKEN")

            embed_model = Model.from_pretrained(
                f"pyannote/{embedding_model}",
                use_auth_token=hf_token,
                strict=False
            )
            inference = Inference(embed_model, window="whole")
            if gpu:
                inference.to(torch.device("cuda"))

            try:
                embedding1 = inference(audio1_path)
                embedding1 = embedding1.reshape(1, -1)

                embedding2 = inference(audio2_path)
                embedding2 = embedding2.reshape(1, -1)

                distance = cdist(embedding1, embedding2, metric="cosine")[0, 0]

                # NOTE: ≤ 0.5: Often considered a “same speaker” indicator.

                results.append({
                    "embedding1": embedding1,
                    "embedding2": embedding2,
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