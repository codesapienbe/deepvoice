# DeepVoice

A comprehensive Python library for voice analysis, processing, and speaker identification.

## Overview

DeepVoice is a powerful toolkit that combines state-of-the-art techniques for voice processing, including speaker diarization, voice verification, and voice embedding representation. It leverages the PyAnnote framework to provide robust tools for audio analysis and voice-related tasks.

## Features

- **Speaker Diarization**: Segment audio by speaker and identify who speaks when
- **Voice Extraction**: Extract individual voice segments from multi-speaker recordings
- **Voice Representation**: Create numerical embeddings of voice characteristics for further analysis
- **Voice Verification**: Compare voice samples to determine if they belong to the same speaker
- **Voice Search**: Find matching voices in a database of voice samples

## Installation

### Requirements

- Python 3.9+
- FFmpeg

### Using pip

```bash
# Install directly from GitHub
pip install git+https://github.com/codesapienbe/deepvoice.git
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/codesapienbe/deepvoice.git
cd deepvoice

# Install in development mode
pip install -e .
```

### Dependencies

The main dependencies will be installed automatically:
```bash
pip install numpy scipy torch torchaudio pyannote.audio librosa soundfile matplotlib PyAudio python-dotenv ffmpeg-python
```

### Hugging Face Token Setup

Many features require a Hugging Face token for accessing pre-trained models:

1. Create an account at [Hugging Face](https://huggingface.co/)
2. Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) to create an access token
3. Accept the user conditions for the models:
   - [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
   - [pyannote/embedding](https://huggingface.co/pyannote/embedding)
4. Set your token as an environment variable:
   ```bash
   export HUGGINGFACE_TOKEN=your_token_here
   ```
   Or create a `.env` file in your project root:
   ```
   HUGGINGFACE_TOKEN=your_token_here
   ```

## Quick Start

```python
from deepvoice import DeepVoice

# Extract voices from an audio file
voices = DeepVoice.extract_voices("recording.wav")
# Output: List of dictionaries with speaker, start time, end time, and path to the extracted voice segment

# Verify if two voice samples belong to the same speaker
verification = DeepVoice.verify("voice1.wav", "voice2.wav")
# Output: List containing dictionary with comparison results, including distance and verification status

# Get a numerical representation of a voice
embeddings = DeepVoice.represent("voice.wav")
# Output: List containing dictionary with voice embeddings

# Find a specific voice in a database of voice recordings
matches = DeepVoice.find("target_voice.wav", "path/to/voice/database")
# Output: List of dictionaries with comparison results for each voice in the database
```

## Advanced Usage

### Speaker Diarization with Custom Parameters

```python
# Extract voices with more speakers and custom model
voices = DeepVoice.extract_voices(
    "recording.wav", 
    model="speaker-diarization-3.0", 
    max_speakers=5
)
```

### Voice Verification with Custom Threshold

```python
# Custom threshold for verification (lower = stricter matching)
verification = DeepVoice.verify(
    "voice1.wav", 
    "voice2.wav", 
    threshold=0.3
)
```

### Working with Voice Embeddings

```python
# Get embeddings from one voice
embeddings1 = DeepVoice.represent("voice1.wav")

# Use the embeddings directly in verification
embeddings2 = DeepVoice.represent("voice2.wav")
verification = DeepVoice.verify(
    embeddings1[0]["embedding"], 
    embeddings2[0]["embedding"]
)
```

## File Structure

The extracted voice segments are saved in `~/.deepvoice/voices/` with filenames that include:
- Session timestamp
- Speaker ID
- Segment index
- Start and end times

This makes it easy to manually inspect and organize the extracted voices.

## Performance Considerations

- Voice processing can be resource-intensive, especially with longer audio files
- Consider using GPU acceleration for faster processing when available
- Process longer audio files in smaller chunks for optimal performance

## License

MIT License - See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use DeepVoice in your research, please cite:

```
@software{deepvoice,
  title = {DeepVoice: A Python Library for Voice Analysis and Processing},
  author = {Yilmaz Mustafa},
  year = {2025},
  url = {https://github.com/codesapienbe/deepvoice}
}
```