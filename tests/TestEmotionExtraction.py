import os
from pathlib import Path
import unittest
from pprint import pprint

from deepvoice import DeepVoice


class TestEmotionExtraction(unittest.TestCase):

    def setUp(self):
        # Get HF token from environment
        self.hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            self.skipTest("HUGGINGFACE_TOKEN environment variable not set")

        # Path to test an audio file
        self.test_audio = "data/trvoice1.wav"
        if not os.path.exists(self.test_audio):
            self.skipTest(f"Test audio file not found: {self.test_audio}")

    def test_extract_emotions(self):
        """Test extracting emotions from a voice recording."""
        print("\nTesting extract_emotions...")

        emotions = DeepVoice.represent_emotions(
            audio_path=self.test_audio,
            hf_token=self.hf_token
        )

        # Check results
        self.assertIsNotNone(emotions)
        self.assertIsInstance(emotions, list)
        if emotions:
            print(f"Found {len(emotions)} emotion classifications:")
            pprint(emotions)

            # Verify structure of results
            self.assertIn("emotion", emotions[0])
            self.assertIn("confidence", emotions[0])
            self.assertIn("path", emotions[0])

            # Check confidence values are reasonable
            self.assertGreaterEqual(emotions[0]["confidence"], 0.0)
            self.assertLessEqual(emotions[0]["confidence"], 1.0)

    def test_analyze_emotional_speech(self):
        """Test analyzing emotions in voice segments."""
        print("\nTesting analyze_emotional_speech...")

        results = DeepVoice.analyze_emotional_speech(
            audio_path=self.test_audio,
            hf_token=self.hf_token,
            max_speakers=2
        )

        # Check results
        self.assertIsNotNone(results)
        self.assertIsInstance(results, list)
        if results:
            print(f"Found {len(results)} voice segments with emotions:")
            for i, segment in enumerate(results[:3]):  # Print first 3 segments
                print(f"\nSegment {i + 1}:")
                print(f"  Speaker: {segment['speaker']}")
                print(f"  Time: {segment['start']}s - {segment['end']}s")
                print(f"  Emotion: {segment['emotion']} (confidence: {segment['emotion_confidence']:.2f})")

            # Verify structure of results
            self.assertIn("speaker", results[0])
            self.assertIn("start", results[0])
            self.assertIn("end", results[0])
            self.assertIn("path", results[0])
            self.assertIn("emotion", results[0])
            self.assertIn("emotion_confidence", results[0])

            # Check confidence values are reasonable
            self.assertGreaterEqual(results[0]["emotion_confidence"], 0.0)
            self.assertLessEqual(results[0]["emotion_confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()