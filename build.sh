#!/bin/bash

# Clean previous builds
rm -rf dist/ build/ *.egg-info/
pip install wheel setuptools python-dotenv numpy scipy torch torchaudio pyannote.audio librosa soundfile matplotlib PyAudio numba onnxruntime

pip install .
