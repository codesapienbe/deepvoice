import os
from typing import Optional, List, Dict, Any

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
            diarization_model="speaker-diarization-3.0",
            hf_token: Optional[str] = None,
            max_speakers: Optional[int] = 3,
            silent: bool = False,
            gpu: Optional[bool] = False
    ) -> List[Dict[str, Any]]:
        try:

            if gpu:
                import torch
                torch.cuda.empty_cache()

            if hf_token is None:
                hf_token = os.getenv("HUGGINGFACE_TOKEN")

            print(f"HF token: {hf_token}")

            pipeline = Pipeline.from_pretrained(
                f"pyannote/{diarization_model}",
                use_auth_token=hf_token
            )

            audio = Audio()
            duration = audio.get_duration(audio_path)

            diarization = pipeline(
                file=audio_path,
                min_speakers=1,
                max_speakers=max_speakers,
                progress=not silent  # Pyannote's built-in progress control
            )

            results = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start = min(turn.start, duration)
                end = min(turn.end, duration)
                if end < start:
                    continue

                try:
                    segment = Segment(start, end)
                    cropped_audio, sample_rate = audio.crop(audio_path, segment)

                    if hasattr(cropped_audio, "numpy"):
                        cropped_audio_np = cropped_audio.numpy()
                    else:
                        cropped_audio_np = np.array(cropped_audio)
                except Exception as e:
                    cropped_audio_np = np.array([])
                    if not silent:
                        print(f"Cropping error for segment {start}-{end}: {str(e)}")

                segment_details = {
                    "speaker": speaker,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "content": cropped_audio_np
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
        """Extract speaker embeddings using pyannote's embedding model."""
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
        """Extract speaker embeddings using pyannote's embedding model."""
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