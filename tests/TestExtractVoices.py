import unittest

from deepvoice import DeepVoice

class TestExtractVoices(unittest.TestCase):

    def setUp(self):
        # Get HF token from environment
        self.hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            self.skipTest("HUGGINGFACE_TOKEN environment variable not set")

        # Path to test an audio file
        self.test_audio = "data/trvoice1.wav"
        if not os.path.exists(self.test_audio):
            self.skipTest(f"Test audio file not found: {self.test_audio}")

def test_extract_voices(audio_path: str):
    voices = DeepVoice.extract_voices(audio_path=audio_path)
    for index, segment in enumerate(voices):
        print(
            f"Segment {index}:\n"
            f"Speaker {segment['speaker']} from {segment['start']}s to {segment['end']}s\n"
            f"Voice path: {segment['path']}"
        )

def test_represent(audio_path: str):
    emb1 = DeepVoice.represent(audio_path)[0]['embedding']
    print(f"Represent: {emb1}")

def test_verify(audio1_path: str, audio2_path: str):
    """ `distance` is a `float` describing how dissimilar speakers 1 and 2 are. """
    verification = DeepVoice.verify(audio1_path, audio2_path)[0]
    print(f"Verify: Distance is {verification['distance']:.2f}, same speaker: {verification['verified']}")

def test_find(audio_path: str, database_path: str):
    """ `distance` is a `float` describing how dissimilar speakers 1 and 2 are. """
    matching_voices = DeepVoice.find(audio_path, database_path)
    for matching_voice in matching_voices:
        print(
            f"Matching voice: Distance is {matching_voice['distance']:.2f}, "
            f"same speaker: {matching_voice['verified']}\n"
            f"Audio 1: {matching_voice['embedding1']}\n"
            f"Audio 2: {matching_voice['embedding2']}"
        )



