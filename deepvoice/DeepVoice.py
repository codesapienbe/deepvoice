import os
from typing import List, Dict, Any, Optional, Union

import librosa
import numpy as np
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
from scipy.spatial import distance

from deepvoice.models import Voice

load_dotenv()


class DeepVoice:
    """
    DeepVoice class for voice processing, diarization, and verification tasks.
    This implementation uses torch, numpy, scipy, and pyannote libraries.
    """
    # Class variables
    # Class variables to hold the state and models
    is_initialized = False
    embedding_model = None
    diarization_pipeline = None
    default_threshold = 0.70
    min_voice_duration = 1.0  # Minimum duration in seconds


    @staticmethod
    def initialize(hf_token: Optional[str] = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Explicitly initialize the DeepVoice system with required models.

        Args:
            hf_token: Hugging Face API token (defaults to environment variable)
            device: Device to use for computation ('cuda' or 'cpu')
        """
        # Check if already initialized
        if DeepVoice.is_initialized:
            print("DeepVoice is already initialized.")
            return

        if hf_token is None:
            # Try to get from environment
            hf_token = os.getenv("HUGGINGFACE_TOKEN")

        if not hf_token:
            raise ValueError(
                "Hugging Face API token is required. Please provide it as an argument or set HUGGINGFACE_TOKEN environment variable.")

        print(f"Initializing DeepVoice on {device}...")

        # Initialize the embedding model for speaker verification
        try:
            DeepVoice.embedding_model = PretrainedSpeakerEmbedding(
                "speechbrain/spkrec-ecapa-voxceleb",
                device=torch.device(device)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize speaker embedding model: {e}")

        # Initialize the diarization pipeline
        try:
            DeepVoice.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            if device == "cuda":
                DeepVoice.diarization_pipeline = DeepVoice.diarization_pipeline.to(torch.device("cuda"))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize diarization pipeline: {e}")

        DeepVoice.is_initialized = True
        print("DeepVoice initialization complete.")

    @staticmethod
    def _ensure_initialized():
        """Ensure the DeepVoice system is initialized or raise an error."""
        if not DeepVoice.is_initialized:
            raise RuntimeError(
                "DeepVoice is not initialized. Call DeepVoice.initialize() first."
            )

    @staticmethod
    def _align_voices(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
        """
        Align voices in audio file and return processed audio.

        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate

        Returns:
            Processed audio as numpy array
        """
        DeepVoice._ensure_initialized()

        # Load audio file
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

        # Basic audio preprocessing
        # Normalize audio
        audio = audio / np.max(np.abs(audio))

        # Noise reduction can be implemented here if needed
        # For simplicity, just return the normalized audio
        return audio

    @staticmethod
    def _calculate_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine distance between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine distance between embeddings
        """
        DeepVoice._ensure_initialized()
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()

        return distance.cosine(embedding1, embedding2)

    @staticmethod
    def _cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine distance between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Cosine distance between embeddings
        """
        DeepVoice._ensure_initialized()
        return distance.cosine(emb1.flatten(), emb2.flatten())

    @staticmethod
    def get_threshold(voices: List[Union[Voice, np.ndarray]], default: float = 0.75) -> float:
        """
        Calculate optimal threshold for voice verification.

        Args:
            voices: List of Voice objects or audio arrays
            default: Default threshold value to return if calculation fails

        Returns:
            Calculated or default threshold value
        """
        DeepVoice._ensure_initialized()

        if len(voices) < 2:
            return default

        try:
            # Extract audio data if Voice objects are provided
            voice_audio = [voice.audio if isinstance(voice, Voice) else voice for voice in voices]

            embeddings = [DeepVoice.represent(audio) for audio in voice_audio]
            distances = []

            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    distances.append(DeepVoice._calculate_distance(embeddings[i], embeddings[j]))

            # Calculate threshold as mean + std of distances
            if distances:
                threshold = np.mean(distances) + np.std(distances)
                return min(threshold, 0.95)  # Cap at 0.95

            return default
        except Exception as e:
            print(f"Error calculating threshold: {e}")
            return default

    @staticmethod
    def extract_voices(audio_path: str, method: str = 'basic') -> List[Voice]:
        """
        Extract individual voices from an audio file.

        Args:
            audio_path: Path to audio file
            method: Method to use for extraction ('basic')

        Returns:
            List of Voice objects representing extracted voice segments
        """
        DeepVoice._ensure_initialized()

        # Load and align audio
        audio = DeepVoice._align_voices(audio_path)
        sr = 16000  # Sample rate

        try:
            # Perform diarization to get voice segments
            segments = DeepVoice._diarize_voices(audio_path)
        except Exception as e:
            print(f"Full diarization failed: {e}")
            try:
                # Try alternative approach with VAD (Voice Activity Detection)
                print("Attempting fallback with basic voice activity detection...")
                segments = DeepVoice._basic_vad_segmentation(audio, sr)
                if segments:
                    # Process the VAD segments similar to diarization segments
                    voices = []
                    for segment in segments:
                        start_sample = int(segment['start'] * sr)
                        end_sample = int(segment['end'] * sr)

                        if start_sample >= len(audio) or end_sample > len(audio):
                            continue

                        voice_segment = audio[start_sample:end_sample]

                        # Only include segments of sufficient length
                        if len(voice_segment) > DeepVoice.min_voice_duration * sr:
                            # VAD segments have lower confidence than full diarization
                            segment_confidence = min(0.7, 0.4 + (len(voice_segment) / (sr * 10)) * 0.3)

                            # Make sure voice_segment is a numpy array
                            if not isinstance(voice_segment, np.ndarray):
                                voice_segment = np.array(voice_segment)

                            voices.append(Voice(
                                audio=voice_segment,  # Must be a numpy array
                                speaker_id=segment['speaker'],
                                start=segment['start'],
                                end=segment['end'],
                                confidence=segment_confidence
                            ))

                    if voices:
                        # If we have multiple voices, try to estimate confidence based on similarity
                        if len(voices) > 1:
                            DeepVoice._refine_voice_confidence(voices)
                        return voices
            except Exception as e2:
                print(f"VAD fallback failed: {e2}")
                print("Falling back to basic voice extraction without diarization")

            # Simple fallback - just use the entire audio as one segment
            if len(audio) > DeepVoice.min_voice_duration * sr:
                duration_sec = len(audio) / sr

                # Ensure audio is a numpy array
                if not isinstance(audio, np.ndarray):
                    audio = np.array(audio)

                return [Voice(
                    audio=audio,  # Must be a numpy array
                    speaker_id="unknown",
                    start=0,
                    end=duration_sec,
                    confidence=0.5  # Default confidence for fallback
                )]
            else:
                return []

        # Extract voice segments from diarization results
        voices = []

        for segment in segments:
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)

            if start_sample >= len(audio) or end_sample > len(audio):
                continue

            voice_segment = audio[start_sample:end_sample]

            # Only include segments of sufficient length
            if len(voice_segment) > DeepVoice.min_voice_duration * sr:
                # Calculate confidence based on segment length - longer segments typically have higher confidence
                segment_confidence = min(0.9, 0.5 + (len(voice_segment) / (sr * 10)) * 0.5)

                # Make sure voice_segment is a numpy array
                if not isinstance(voice_segment, np.ndarray):
                    voice_segment = np.array(voice_segment)

                voices.append(Voice(
                    audio=voice_segment,  # Must be a numpy array
                    speaker_id=segment['speaker'],
                    start=segment['start'],
                    end=segment['end'],
                    confidence=segment_confidence
                ))

        # If we have multiple voices, try to estimate confidence based on similarity within speaker groups
        if len(voices) > 1:
            DeepVoice._refine_voice_confidence(voices)

        return voices

    @staticmethod
    def _basic_vad_segmentation(audio: np.ndarray, sr: int = 16000) -> List[Dict[str, Any]]:
        """
        Perform basic voice activity detection to segment audio.
        """
        import librosa

        try:
            # Parameters for VAD
            frame_length = int(0.025 * sr)  # 25ms frame
            hop_length = int(0.010 * sr)  # 10ms hop
            n_fft = 2048

            # Log input audio stats
            print(f"Audio shape: {audio.shape}, Sample rate: {sr}")
            print(f"Audio min: {np.min(audio)}, max: {np.max(audio)}, mean: {np.mean(audio)}")

            # Energy-based VAD
            # Compute frame energy
            energy = np.array([
                np.sum(audio[i:i + frame_length] ** 2)
                for i in range(0, len(audio) - frame_length, hop_length)
            ])

            # Compute spectral features
            spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
            spec_sum = np.sum(spec, axis=0)

            print(f"Energy shape: {energy.shape}, Spec sum shape: {spec_sum.shape}")

            # Normalize
            energy_norm = energy / np.max(energy) if np.max(energy) > 0 else energy
            spec_norm = spec_sum / np.max(spec_sum) if np.max(spec_sum) > 0 else spec_sum

            # Make sure both arrays have the same length
            min_length = min(len(energy_norm), len(spec_norm))
            energy_norm = energy_norm[:min_length]
            spec_norm = spec_norm[:min_length]

            print(f"After truncation - Energy norm shape: {energy_norm.shape}, Spec norm shape: {spec_norm.shape}")

            # Combined feature (both energy and spectral content)
            feature = energy_norm * spec_norm

            # Threshold for speech detection
            # Use dynamic thresholding - percentile based
            threshold = np.percentile(feature, 20)  # Adaptive threshold

            # Find segments
            is_speech = feature > threshold

            # Convert frame-level decisions to segments
            segments = []
            in_segment = False
            current_start = 0

            for i, speech in enumerate(is_speech):
                frame_time = i * hop_length / sr

                if speech and not in_segment:
                    # Start of a new segment
                    in_segment = True
                    current_start = frame_time
                elif not speech and in_segment:
                    # End of a segment
                    in_segment = False
                    segment_end = frame_time

                    # Only keep segments of reasonable length (not too short)
                    if segment_end - current_start >= DeepVoice.min_voice_duration:
                        segments.append({
                            "start": current_start,
                            "end": segment_end,
                            "speaker": f"vad_speaker_{len(segments)}",  # Simple speaker labeling
                        })

            # If still in a segment at the end of the audio
            if in_segment:
                segment_end = len(audio) / sr
                if segment_end - current_start >= DeepVoice.min_voice_duration:
                    segments.append({
                        "start": current_start,
                        "end": segment_end,
                        "speaker": f"vad_speaker_{len(segments)}",
                    })

            # Merge segments that are very close together (less than 0.5s apart)
            if segments:
                merged_segments = [segments[0]]
                for segment in segments[1:]:
                    prev_segment = merged_segments[-1]

                    # If this segment starts soon after the previous one ends
                    if segment["start"] - prev_segment["end"] < 0.5:
                        # Merge by extending the previous segment
                        prev_segment["end"] = segment["end"]
                    else:
                        # Add as a new segment
                        merged_segments.append(segment)

                return merged_segments

            return segments

        except Exception as e:
            print(f"Detailed VAD error: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise


    @staticmethod
    def _refine_voice_confidence(voices: List[Voice]) -> None:
        """
        Refine confidence scores for a list of Voice objects.
        Updates the confidence attribute in-place.

        Args:
            voices: List of Voice objects to update
        """
        # Group voices by speaker_id
        speakers = {}
        for voice in voices:
            if voice.speaker_id not in speakers:
                speakers[voice.speaker_id] = []
            speakers[voice.speaker_id].append(voice)

        # For each speaker group with multiple segments, calculate confidence based on internal similarity
        for speaker_id, speaker_voices in speakers.items():
            if len(speaker_voices) > 1:
                # Calculate embeddings for all voices in this speaker group
                embeddings = [DeepVoice.represent(voice.audio) for voice in speaker_voices]

                # Calculate average pairwise distance
                for i, voice in enumerate(speaker_voices):
                    distances = []
                    for j, embedding in enumerate(embeddings):
                        if i != j:  # Skip comparing with self
                            distances.append(DeepVoice._calculate_distance(
                                DeepVoice.represent(voice.audio), embedding))

                    if distances:
                        avg_distance = np.mean(distances)
                        # Lower distance = higher confidence
                        confidence = max(voice.confidence, 1.0 - min(avg_distance, 0.9))
                        voice.confidence = confidence

    @staticmethod
    def verify(voice1: Union[Voice, np.ndarray], voice2: Union[Voice, np.ndarray],
               threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Verify if two voice segments belong to the same speaker.

        Args:
            voice1: First voice (Voice object or audio array)
            voice2: Second voice (Voice object or audio array)
            threshold: Similarity threshold (lower distance = more similar)

        Returns:
            Dictionary with verification results
        """
        DeepVoice._ensure_initialized()

        if threshold is None:
            threshold = DeepVoice.default_threshold

        # Extract audio data if Voice objects are provided
        audio1 = voice1.audio if isinstance(voice1, Voice) else voice1
        audio2 = voice2.audio if isinstance(voice2, Voice) else voice2

        embedding1 = DeepVoice.represent(audio1)
        embedding2 = DeepVoice.represent(audio2)

        distance_score = DeepVoice._calculate_distance(embedding1, embedding2)
        is_same_speaker = distance_score < threshold

        confidence = 1.0 - (distance_score / threshold) if is_same_speaker else distance_score / 2
        confidence = max(0.0, min(1.0, confidence))

        return {
            "is_same": is_same_speaker,
            "distance": distance_score,
            "threshold": threshold,
            "confidence": confidence
        }

    @staticmethod
    def find(target_voice: Union[Voice, np.ndarray], voice_list: List[Union[Voice, np.ndarray]],
             threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Find instances of the target voice in a list of voice segments.

        Args:
            target_voice: Target voice to find (Voice object or audio array)
            voice_list: List of voices to search in (Voice objects or audio arrays)
            threshold: Similarity threshold

        Returns:
            List of matches with their confidence scores
        """
        DeepVoice._ensure_initialized()

        from tqdm import tqdm

        if threshold is None:
            threshold = DeepVoice.default_threshold

        # Extract audio data if Voice object is provided
        target_audio = target_voice.audio if isinstance(target_voice, Voice) else target_voice
        target_embedding = DeepVoice.represent(target_audio)

        results = []

        # Add progress bar
        total_voices = len(voice_list)
        print(f"Comparing target voice to {total_voices} voice samples...")

        for i, voice in tqdm(enumerate(voice_list), total=total_voices, desc="Voice lookup progress"):
            # Extract audio data if Voice object is provided
            audio = voice.audio if isinstance(voice, Voice) else voice
            embedding = DeepVoice.represent(audio)

            distance_score = DeepVoice._calculate_distance(target_embedding, embedding)
            is_match = distance_score < threshold

            # Calculate match confidence
            match_confidence = 1.0 - (distance_score / threshold) if is_match else 0.0
            match_confidence = max(0.0, min(1.0, match_confidence))

            # If voice is a Voice object, incorporate its intrinsic confidence
            if isinstance(voice, Voice):
                match_confidence = match_confidence * voice.confidence

            if is_match:
                # If voice is a Voice object, create a new one with updated confidence
                if isinstance(voice, Voice):
                    # Create a copy with updated confidence
                    result_voice = Voice(
                        audio=voice.audio,
                        speaker_id=voice.speaker_id,
                        start=voice.start,
                        end=voice.end,
                        confidence=match_confidence
                    )
                else:
                    # For numpy arrays, just keep the original
                    result_voice = voice

                results.append({
                    "index": i,
                    "voice": result_voice,
                    "distance": distance_score,
                    "confidence": match_confidence
                })

        # Sort results by confidence (highest first)
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    @staticmethod
    def represent(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Create an embedding representation of a voice audio segment.

        Args:
            audio: Audio waveform as numpy array
            sr: Sample rate, defaults to 16000

        Returns:
            Embedding vector as 1D numpy array
        """
        DeepVoice._ensure_initialized()

        # Ensure audio is a numpy array with correct data type
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)

        # Normalize audio if needed
        if np.max(np.abs(audio)) > 0.0 and np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))

        # Prepare the audio tensor with the correct 3D shape (batch, channel, samples)
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        elif len(audio_tensor.shape) == 2:
            audio_tensor = audio_tensor.unsqueeze(1)

        # Get the embedding
        with torch.no_grad():
            embedding = DeepVoice.embedding_model(audio_tensor)

        # Convert to numpy array if it's a tensor
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()

        # Ensure the embedding is 1D for distance calculation
        return embedding.flatten()  # Convert to 1D vector


    @staticmethod
    def _calculate_voice_confidence(voice: Union[Voice, np.ndarray],
                                    reference_voices: List[Union[Voice, np.ndarray]]) -> float:
        """
        Calculate confidence score for a voice segment based on reference voices.

        Args:
            voice: Voice to evaluate (Voice object or audio array)
            reference_voices: List of reference voices (Voice objects or audio arrays)

        Returns:
            Confidence score between 0.0 and 1.0
        """
        DeepVoice._ensure_initialized()

        if not reference_voices:
            return 0.5  # No reference, default confidence

        # Extract audio data if Voice objects are provided
        voice_audio = voice.audio if isinstance(voice, Voice) else voice
        ref_audio_list = [v.audio if isinstance(v, Voice) else v for v in reference_voices]

        voice_embedding = DeepVoice.represent(voice_audio)
        distances = []

        for ref_audio in ref_audio_list:
            ref_embedding = DeepVoice.represent(ref_audio)
            distances.append(DeepVoice._calculate_distance(voice_embedding, ref_embedding))

        # Lower distance = higher confidence
        avg_distance = np.mean(distances)
        confidence = 1.0 - min(avg_distance, 1.0)

        return confidence


    @staticmethod
    def _diarize_voices(audio_path: str) -> List[Dict[str, Any]]:
        """
        Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            List of diarization segments
        """
        DeepVoice._ensure_initialized()

        if DeepVoice.diarization_pipeline is None:
            raise RuntimeError(
                "Diarization pipeline not initialized. This is likely because:\n"
                "1. You need to set the HF_TOKEN environment variable with your Hugging Face token\n"
                "2. There was a network error while downloading the model\n"
                "3. You need to accept the model's usage conditions on Hugging Face\n\n"
                "To set the token: export HF_TOKEN='your_huggingface_token'\n"
                "You can get a token at https://huggingface.co/settings/tokens"
            )

        # Perform diarization
        diarization = DeepVoice.diarization_pipeline(audio_path)

        # Extract segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })

        # Sort segments by start time
        segments.sort(key=lambda x: x["start"])
        return segments

