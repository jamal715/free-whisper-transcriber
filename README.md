# Free Whisper Transcriber

A reusable Streamlit app for transcribing audio and video with **faster-whisper**.

## What it does

- Upload MP4, MP3, WAV, M4A, WEBM, OGG, FLAC and related media.
- Transcribe locally/on the Streamlit server with Whisper.
- Auto-detect language, including Urdu and English.
- Optionally translate speech into English.
- Show timestamps.
- Download plain-text transcript and SRT subtitles.
- Produce a simple extractive summary without a paid LLM/API.

## Why this is free

The transcription engine is the open-source Whisper model running through
`faster-whisper`. No OpenAI API key or paid transcription service is required.

Hosting on Streamlit Community Cloud can also be used without paying for an API,
subject to Streamlit's current free-tier resource limits.

## Recommended model

Start with **Base** for CPU use.

- Tiny: fastest, least accurate
- Base: best default
- Small: better accuracy, more CPU/RAM/time

For long recordings on free cloud compute, Base is usually the practical choice.

## Run locally

### 1. Clone/download the repository

Open a terminal in the project folder.

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Start the app

```bash
streamlit run app.py
```

Your browser will open the upload interface.

## Deploy on Streamlit Community Cloud

1. Put this repository on GitHub.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Choose **Create app**.
4. Select this repository.
5. Set the entry point to `app.py`.
6. Deploy.

No secrets or API keys are needed.

## Privacy

The app writes an uploaded recording to a temporary file only while it is being
processed, then deletes that temporary file. If deployed publicly, anyone who
can access the Streamlit URL can upload a recording and use the app's compute.
For sensitive recordings, run the app locally or restrict access through your
hosting setup.

## Important practical note

Whisper inference is compute-heavy. A 45-minute recording can be slow on free
shared Streamlit CPU, especially with the Small model. The same app usually runs
more predictably on your own laptop.

## Project structure

```text
free-whisper-transcriber/
├── app.py
├── transcriber.py
├── summarizer.py
├── utils.py
├── requirements.txt
├── packages.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
└── README.md
```
