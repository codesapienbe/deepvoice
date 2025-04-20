import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate


class Voice:
    def __init__(self, audio: np.ndarray, speaker_id: str, start: float, end: float, confidence: float = 0.5):
        self.audio = audio
        self.speaker_id = speaker_id
        self.start = start
        self.end = end
        self.duration = end - start
        self.confidence = confidence  # Add confidence attribute with default value

    def __len__(self):
        return len(self.audio)

