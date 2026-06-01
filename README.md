# Polygot Minutes

AI-powered meeting transcription and summarization — runs 100% locally, no API keys required.

Upload any audio or video recording and get back a structured report: meeting title, key points, narrative summary, decisions made, and action items with priority levels.

## Stack

- **FastAPI** — async REST API with auto-generated OpenAPI docs
- **faster-whisper** — on-device speech-to-text, 2-4x faster than standard Whisper
- **Ollama + Llama 3.2** — local LLM for AI analysis, no internet required
- **Vanilla JS frontend** — served directly from FastAPI, no build step

## Setup

### 1. Install Ollama

Download from [ollama.com](https://ollama.com) and open the app, then pull the model:

```bash
ollama pull llama3.2
```

### 2. Install dependencies

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

The defaults work out of the box — no keys to add:

```
WHISPER_MODEL_SIZE=small
OLLAMA_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.2
MAX_FILE_SIZE_MB=200
PORT=8001
```

### 4. Run

```bash
python run.py
```

Open `http://localhost:8001` — the frontend loads automatically.

Interactive API docs: `http://localhost:8001/docs`

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Server status and model info |
| `POST` | `/api/v1/transcribe` | Transcribe audio/video → text + segments |
| `POST` | `/api/v1/analyze` | Analyze a transcript with Ollama |
| `POST` | `/api/v1/process` | Full pipeline: transcribe + analyze |
| `POST` | `/api/v1/export/markdown` | Full pipeline → download `.md` file |

### Example: transcribe then analyze

```bash
# Step 1 — transcribe
curl -X POST http://localhost:8001/api/v1/transcribe \
  -F "file=@meeting.mp4" | python -m json.tool

# Step 2 — analyze
curl -X POST http://localhost:8001/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"transcript": "your transcript text here"}' | python -m json.tool
```

Response shape from `/api/v1/process`:

```json
{
  "transcript": "...",
  "language": "en",
  "duration": 1842.5,
  "title": "Q3 Product Roadmap Review",
  "key_points": ["..."],
  "summary": "...",
  "decisions": ["Launch feature X by end of Q3"],
  "action_items": [
    { "item": "Draft technical spec", "priority": "High", "assignee": "Alex" }
  ],
  "segments": [{ "start": 0.0, "end": 4.2, "text": "..." }]
}
```

---

## Supported Formats

Audio: `mp3` `wav` `m4a` `ogg` `flac` `opus`  
Video: `mp4` `mov` `webm` `avi` `mkv`

---

## Project Structure

```
app/
├── config.py              # Pydantic Settings (reads .env)
├── models.py              # Request/response schemas
├── main.py                # FastAPI app factory + lifespan
├── routers/
│   └── meetings.py        # All API endpoints
└── services/
    ├── transcription.py   # faster-whisper (async, thread-pool)
    └── summarization.py   # Ollama via OpenAI-compatible API
static/
└── index.html             # Frontend (served by FastAPI)
run.py                     # Entry point
```

## Notes

- Everything runs locally — no audio or text is ever sent to an external service
- faster-whisper model weights are downloaded automatically on first run (~500 MB for `small`)
- For better analysis quality, switch to a larger model: `ollama pull qwen2.5:7b` and set `OLLAMA_MODEL=qwen2.5:7b` in `.env`
- Do not commit `.env` — it is already in `.gitignore`

---

## Troubleshooting

**"Cannot reach Ollama"**  
Open the Ollama app or run `ollama serve` in a terminal before starting the server.

**"Model not found in Ollama"**  
Run `ollama pull llama3.2` (or whichever model is set in `OLLAMA_MODEL`).

**Transcription is slow**  
Switch to a smaller model: set `WHISPER_MODEL_SIZE=tiny` in `.env`. Accuracy drops slightly but speed improves significantly.

**Analysis output is malformed or incomplete**  
Try a more capable model: `ollama pull qwen2.5:7b` and update `OLLAMA_MODEL` in `.env`.

**Port already in use**  
Change `PORT=8002` in `.env`, then open `http://localhost:8002`.

---

## Model Size Guide

| Whisper Model | Speed | Accuracy | RAM |
|---------------|-------|----------|-----|
| `tiny`   | fastest | lower  | ~1 GB |
| `small`  | fast    | good   | ~2 GB |
| `medium` | slower  | better | ~5 GB |

| Ollama Model | Quality | RAM |
|--------------|---------|-----|
| `llama3.2` (3B) | good for demos | ~3 GB |
| `qwen2.5:7b`    | better structured output | ~5 GB |
