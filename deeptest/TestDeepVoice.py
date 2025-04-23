from deepvoice import DeepVoice

def test_extract_voices(audio_path: str):
    voices = DeepVoice.extract_voices(audio_path=audio_path)
    for segment in voices:
        print(
            f"Speaker {segment['speaker']} from {segment['start']}s to {segment['end']}s, "
            f"audio shape: {segment['content'].shape}"
        )

def test_represent(audio_path: str):
    emb1 = DeepVoice.represent(audio_path)[0]['embedding']
    print(f"Represent: {emb1}")

def test_verify(audio1_path: str, audio2_path: str):
    """ `distance` is a `float` describing how dissimilar speakers 1 and 2 are. """
    verification = DeepVoice.verify(audio1_path, audio2_path)[0]
    print(f"Verify: Distance is {verification['distance']:.2f}, same speaker: {verification['verified']}")


if __name__ == "__main__":
    # test_extract_voices("aliseriati.wav")
    # test_represent("trvoice1.wav")
    test_verify("trvoice1.wav", "trvoice2.wav")


