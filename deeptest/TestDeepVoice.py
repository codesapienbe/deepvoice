from deepvoice import DeepVoice

def test_extract_voices(audio_path: str):
    voices = DeepVoice.extract_voices(audio_path=audio_path)
    for segment in voices:
        print(
            f"Speaker {segment['speaker']} from {segment['start']}s to {segment['end']}s, "
            f"audio shape: {segment['content'].shape}"
        )


if __name__ == "__main__":
    test_extract_voices("tedtalk.wav")

