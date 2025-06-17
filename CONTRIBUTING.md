# Contributing New Backends

To add a new backend for DeepVoice features, follow these steps:

1. Implement a subclass of the abstract base classes in `src/deepvoice/backends/base.py`:
   - `DiarizationBackend` for speaker diarization
   - `EmbeddingBackend` for generating embeddings
   - `VerificationBackend` for comparing two voice samples

2. Place your implementation in a new module under `src/deepvoice/backends/`. For example:
   ```python
   # src/deepvoice/backends/mybackend.py
   from deepvoice.backends.base import DiarizationBackend

   class MyDiarization(DiarizationBackend):
       def __init__(self, ...):
           ...
       def diarize(self, audio_path: str, **kwargs) -> List[Dict]:
           ...
   ```

3. Import and inject your backend when instantiating `DeepVoice`:
   ```python
   from deepvoice.backends.mybackend import MyDiarization
   dv = DeepVoice(diarization_backend=MyDiarization(...))
   segments = dv.diarize("audio.wav")
   ```

4. Write unit tests to ensure your backend implements the contract:
   - `diarize` must return a `List[Dict]` with keys `speaker`, `start`, `end`, and `path`.
   - `embed` must return a `numpy.ndarray`.
   - `verify` must return a `float` distance value.

5. Run existing tests and add coverage for your backend. 