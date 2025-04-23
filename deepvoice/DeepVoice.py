import os
from typing import Optional, List, Dict, Any

from pyannote.audio import Pipeline, Audio
from pyannote.core import Segment
import numpy as np

class DeepVoice:

    @staticmethod
    def extract_voices(
        audio_path: Any,
        diarization_model="speaker-diarization-3.0",
        hf_token: Optional[str] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        try:
            if hf_token is None:
                hf_token = os.getenv("HUGGINGFACE_TOKEN")

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
            )

            results = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start = min(turn.start, duration)
                end = min(turn.end, duration)
                if end > start:
                    try:
                        segment = Segment(start, end)
                        cropped_audio, sample_rate = audio.crop(audio_path, segment)
                        # Ensure it's a numpy array (PyTorch Tensor to numpy)
                        if hasattr(cropped_audio, "numpy"):
                            cropped_audio_np = cropped_audio.numpy()
                        else:
                            cropped_audio_np = np.array(cropped_audio)
                    except Exception as e:
                        cropped_audio_np = np.array([])
                        print(f"Cropping error for segment {start}-{end}: {str(e)}")

                    segment_details = {
                        "speaker": speaker,
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "content": cropped_audio_np  # NumPy array
                    }
                    results.append(segment_details)

            return results

        except Exception as e:
            print(f"Voice processing error: {str(e)}")
            return []
