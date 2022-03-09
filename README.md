This is a Python module for Vosk.

Vosk is an offline open source speech recognition toolkit. It enables
speech recognition for 20+ languages and dialects - English, Indian
English, German, French, Spanish, Portuguese, Chinese, Russian, Turkish,
Vietnamese, Italian, Dutch, Catalan, Arabic, Greek, Farsi, Filipino,
Ukrainian, Kazakh, Swedish, Japanese, Esperanto. More to come.

Vosk models are small (50 Mb) but provide continuous large vocabulary
transcription, zero-latency response with streaming API, reconfigurable
vocabulary and speaker identification.

Vosk supplies speech recognition for chatbots, smart home appliances,
virtual assistants. It can also create subtitles for movies,
transcription for lectures and interviews.

Vosk scales from small devices like Raspberry Pi or Android smartphone to
big clusters.

# Documentation

For installation instructions, examples and documentation visit [Vosk
Website](https://alphacephei.com/vosk). See also our project on
[Github](https://github.com/alphacep/vosk-api).

# Installation Guideline
1. Install python 3.7.0
2. Create and activate a virtual environment for the project
	> Command:
```
	python -m venv (projectname)
	(projectname)\Scripts\activate.bat
```
3. Run the requirements.txt to install packages
```
	pip install -r requirements.txt
```
4. For STT, please copy the model to 'main' folder, and rename it to 'model'
5. In 'main' folder, run the script to call the model and test:
```
	python ffmpeg.py (test audio file name.wav/mp3)
	 e.g: python ffmpeg.py welcome.wav
```