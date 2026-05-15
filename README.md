# Polyglot Minutes

AI-powered meeting transcription and summarization. Upload any audio or video recording and get back a structured, AI-generated report: title, key points, narrative summary, decisions made, and action items with priority levels.

## Stack

- **FastAPI** — async REST API with auto-generated OpenAPI docs
- **OpenAI Whisper** — local on-device speech-to-text (no audio leaves your machine)
- **OpenAI GPT-4o-mini** — AI analysis: title, summary, decisions, action items
- **Vanilla JS frontend** — served directly from FastAPI, no build step

## Setup

### 1. Install dependencies

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

```
OPENAI_API_KEY=sk-...
WHISPER_MODEL_SIZE=small   # tiny | base | small | medium | large
MAX_FILE_SIZE_MB=200
PORT=8001
```

### 3. Run

```bash
python run.py
```

Open `http://localhost:8001` — the frontend loads automatically.

Interactive API docs: `http://localhost:8001/docs`

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Server status, model info, AI availability |
| `POST` | `/api/v1/transcribe` | Transcribe audio/video → text + segments |
| `POST` | `/api/v1/analyze` | Analyze a transcript text with GPT |
| `POST` | `/api/v1/process` | Full pipeline: transcribe + analyze |
| `POST` | `/api/v1/export/markdown` | Full pipeline → download `.md` file |

### Example: Full pipeline

```bash
curl -X POST http://localhost:8001/api/v1/process \
  -F "file=@meeting.mp4" | python -m json.tool
```

Response shape:

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
├── config.py          # Pydantic Settings (reads .env)
├── models.py          # Request/response schemas
├── main.py            # FastAPI app factory + lifespan
├── routers/
│   └── meetings.py    # All API endpoints
└── services/
    ├── transcription.py   # Whisper (thread-pool, async-safe)
    └── summarization.py   # OpenAI GPT-4o-mini analysis
static/
└── index.html         # Frontend (served by FastAPI)
run.py                 # Entry point
```

## Notes

- Whisper runs locally — audio never leaves your machine
- GPT-4o-mini is used for analysis only (transcript text is sent to OpenAI)
- Whisper model weights are downloaded automatically on first run (~140 MB for `small`)
- Do not commit `.pt`/`.pth` files or `.env` — both are in `.gitignore`
