# End-to-end Text to Speech

End-to-end processing engine from website text to speech audio output.

Features:
* Extract main text from websites using trafilatura
* Preprocess text with NVIDIA NeMO and some custom code
* Split text into sentences, taking maximum number of tokens into account
* Generate speech with Coqui XTTS-v2
* Validate speech samples with whisper-timestamped, regenerating sample if necessary
* Concat sentences into one WAV
* Enhance WAV with noisereduce

# Setup
Install the following dependencies:
`pip install noisereduce requests beautifulsoup4 trafilatura nemo_toolkit[all] whisper-timestamped TTS`

# License
Due to the dependency on whisper-timestamped, the whole project is licensed under APGL-3.0.
