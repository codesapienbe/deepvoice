import os
from pathlib import Path
import unittest
from pprint import pprint
import numpy as np

from deepvoice import DeepVoice


class TestDeepVoice(unittest.TestCase):

    def setUp(self):
        self.hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            self.skipTest("HUGGINGFACE_TOKEN environment variable not set")

        self.test_audio = "data/trvoice1.wav"
        self.ref_audio = "data/trvoice2.wav"
        self.db_path = "data/voice_db"

        if not os.path.exists(self.test_audio):
            self.skipTest(f"Test audio file not found: {self.test_audio}")

    # Helper method
    def assertValidSegment(self, segment):
        self.assertIn("speaker", segment)
        self.assertIn("start", segment)
        self.assertIn("end", segment)
        self.assertIn("path", segment)
        self.assertTrue(os.path.exists(segment["path"]))

    ### Test Cases ###

    def test_extract_voices(self):
        """Test speaker diarization and voice extraction"""
        print("\nTesting extract_voices...")

        segments = DeepVoice.extract_voices(
            self.test_audio,
            hf_token=self.hf_token,
            max_speakers=2
        )

        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)
        self.assertValidSegment(segments[0])
        print(f"Extracted {len(segments)} voice segments")

    def test_represent_voice(self):
        """Test voice embedding extraction"""
        print("\nTesting represent_voice...")

        embeddings = DeepVoice.represent_voice(
            self.test_audio,
            hf_token=self.hf_token
        )

        self.assertIsInstance(embeddings, list)
        self.assertGreater(len(embeddings), 0)
        self.assertIn("embedding", embeddings[0])
        self.assertIsInstance(embeddings[0]["embedding"], np.ndarray)
        print(f"Embedding shape: {embeddings[0]['embedding'].shape}")

    def test_verify_voice(self):
        """Test voice verification functionality"""
        print("\nTesting verify_voice...")

        # Same speaker test
        same_result = DeepVoice.verify_voice(
            self.test_audio,
            self.test_audio,
            hf_token=self.hf_token
        )[0]

        self.assertTrue(same_result["verified"])
        self.assertLess(same_result["distance"], 0.5)

        # Different speaker test
        diff_result = DeepVoice.verify_voice(
            self.test_audio,
            self.ref_audio,
            hf_token=self.hf_token
        )[0]

        self.assertFalse(diff_result["verified"])
        print(f"Same speaker distance: {same_result['distance']:.3f}")
        print(f"Different speaker distance: {diff_result['distance']:.3f}")

    def test_find_voices(self):
        """Test voice database search"""
        print("\nTesting find_voices...")

        if not os.path.exists(self.db_path):
            self.skipTest(f"Voice database not found: {self.db_path}")

        results = DeepVoice.find_voices(
            self.test_audio,
            self.db_path,
            hf_token=self.hf_token
        )

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIn("verified", results[0])
        print(f"Found {len(results)} potential matches")

        # Check at least one match in database
        matches = [r for r in results if r["verified"]]
        self.assertGreater(len(matches), 0)
        print(f"Verified matches: {len(matches)}")

    def test_represent_emotions(self):
        """Test emotion recognition in audio"""
        print("\nTesting represent_emotions...")

        emotions = DeepVoice.represent_emotions(
            self.test_audio,
            hf_token=self.hf_token
        )

        self.assertIsInstance(emotions, list)
        self.assertGreater(len(emotions), 0)
        self.assertIn("emotion", emotions[0])
        self.assertIn("confidence", emotions[0])
        print(f"Detected emotion: {emotions[0]['emotion']} ({emotions[0]['confidence']:.2f})")

    def test_extract_emotions(self):
        """Test emotion-aware voice extraction"""
        print("\nTesting extract_emotions...")

        results = DeepVoice.extract_emotions(
            self.test_audio,
            hf_token=self.hf_token
        )

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertValidSegment(results[0])
        self.assertIn("emotion", results[0])
        self.assertIn("emotion_confidence", results[0])
        print(f"First segment emotion: {results[0]['emotion']}")


if __name__ == "__main__":
    unittest.main()
