import os

from deepvoice import DeepVoice


def deepvoice_test():
    """Example of using the DeepVoice API"""
    # Initialize
    DeepVoice.initialize()

    audio_path: str = os.path.join(os.getcwd(), "deeptest", "audio", "tedtalk.mp3")

    # Extract voices from audio file
    voices = DeepVoice.extract_voices(audio_path)

    # Print information about extracted voices
    print(f"Found {len(voices)} voice segments:")
    for i, voice in enumerate(voices):
        print(
            f"  Voice {i + 1}: Speaker {voice.speaker_id}, Duration: {voice.duration:.2f}s, Confidence: {voice.confidence:.2f}")

    # If we have multiple voices, verify if they're from the same speaker
    if len(voices) >= 2:
        result = DeepVoice.verify(voices[0], voices[1])
        print(f"Verification: Same speaker? {result['is_same']} (confidence: {result['confidence']:.2f})")

    # Find similar voices
    if len(voices) >= 3:
        matches = DeepVoice.find(voices[0], voices[1:])
        print(f"Found {len(matches)} matches for the first voice")
        for match in matches:
            voice = match["voice"]
            print(f"  Match: Speaker {voice.speaker_id}, Confidence: {match['confidence']:.2f}")


if __name__ == "__main__":
    deepvoice_test()