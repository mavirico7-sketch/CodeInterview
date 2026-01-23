# Interview Simulator

AI-powered technical interview simulator using LLM (OpenAI API). This service simulates a technical interview with a focus on the question-and-answer phase.

## Features

- **Dynamic Question Planning**: AI generates and adapts questions based on position, tech stack, and candidate level
- **Session Resumption**: Get full session state including messages to resume interrupted interviews
- **Context Summarization**: Automatically summarizes older chat messages when context exceeds token limit

## Quick Start

1. **Clone and configure**:
   ```bash
   cp interview.conf.example interview.conf
   nano interview.conf  # Set your OPENAI_API_KEY
   ```

2. **Start the services**:
   ```bash
   docker-compose up -d
   ```

3. **Verify the service is running**:
   ```bash
   curl http://localhost:8001/api/v1/health
   ```

## Configuration

All configuration is in `interview.conf`:

```bash
# OpenAI API Settings
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=4096

# MongoDB Settings
MONGODB_HOST=mongodb
MONGODB_PORT=27017
MONGODB_DATABASE=interview_simulator

# Interview Settings
INTERVIEW_CONTEXT_TOKEN_LIMIT=50000
INTERVIEW_MAX_EXCHANGES=25
MAX_CHALLENGES=3
```

Prompts are configured separately in `app/prompts.yaml`.

## API Reference

Base URL: `http://localhost:8001/api/v1`

### Health Check

```bash
curl http://localhost:8001/api/v1/health
```

### Create Session

Creates a new interview session with generated question plan.

```bash
curl -X POST http://localhost:8001/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "vacancy": "Backend Developer",
    "stack": "Python, FastAPI, PostgreSQL, Redis",
    "level": "middle",
    "language": "Russian"
  }'
```

**Request body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `vacancy` | string | yes | Job position title |
| `stack` | string | yes | Technology stack (comma-separated) |
| `level` | string | yes | Candidate level: `intern`, `junior`, `middle`, `senior`, `lead`, `principal` |
| `language` | string | no | Interview language (default: "English") |

**Response:**
```json
{
  "session_id": "6789abc...",
  "phase": "interview",
  "created_at": "2025-01-12T10:00:00Z",
  "is_active": true,
  "exchange_count": 0,
  "total_tokens_used": 1500
}
```

### Get Session

Returns full session state including messages for session resumption.

```bash
curl http://localhost:8001/api/v1/sessions/{session_id}
```

**Response:**
```json
{
  "session_id": "6789abc...",
  "phase": "interview",
  "created_at": "2025-01-12T10:00:00Z",
  "is_active": true,
  "exchange_count": 3,
  "total_tokens_used": 15000,
  "init_info": {
    "vacancy": "Backend Developer",
    "stack": "Python, FastAPI, PostgreSQL, Redis",
    "level": "middle",
    "language": "Russian"
  },
  "display_messages": {
    "interview": [
      {
        "role": "assistant",
        "content": "Здравствуйте! Меня зовут...",
        "timestamp": "2025-01-12T10:01:00Z"
      },
      {
        "role": "user",
        "content": "Привет! Я работаю...",
        "timestamp": "2025-01-12T10:02:00Z"
      }
    ],
    "live_coding": [],
    "final": []
  }
}
```

### Start Interview

Starts the interview. The AI interviewer will introduce themselves and ask the first question. Returns a single JSON response.

```bash
curl -X POST http://localhost:8001/api/v1/sessions/{session_id}/start
```

**Errors:**
- `404` — Session not found
- `400` — Interview already started

### Send Message

Send a candidate's response. The AI will process the answer and ask the next question. Returns a single JSON response.

```bash
curl -X POST http://localhost:8001/api/v1/sessions/{session_id}/message \
  -H "Content-Type: application/json" \
  -d '{"message": "I have 3 years of experience with Python...", "current_code": ""}'
```

**Errors:**
- `404` — Session not found
- `400` — Session is no longer active
- `400` — Question phase is complete

## Response Format

Responses from `/start` and `/message` endpoints are returned as JSON.

**Example response:**
```json
{
  "session_id": "6789abc...",
  "content": "Hello! My name is Alex...",
  "phase": "interview",
  "exchange_count": 1,
  "total_tokens_used": 3200,
  "phase_changed": false,
  "is_phase_complete": false
}
```

## How To Detect Phase Transitions

Use the `phase_changed` flag and `phase` field in the JSON response:

- `phase_changed: true` means the phase advanced during this request.
- `phase` contains the current phase (`interview`, `live_coding`, `final`).

Tool calls are not exposed to the client directly.

## Interview Phases

| Phase | Description |
|-------|-------------|
| `interview` | Q&A phase — AI asks questions, candidate answers |
| `live_coding` | Live coding phase |
| `final` | Final summary phase |

## Display Messages By Phase

In session responses, `display_messages` are grouped by phase:

```json
{
  "display_messages": {
    "interview": [/* messages from interview phase */],
    "live_coding": [/* messages from live coding phase */],
    "final": [/* messages from final phase */]
  }
}
```

## Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Export config
export $(grep -v '^#' interview.conf | xargs)

# Start MongoDB
docker run -d -p 27017:27017 mongo:7.0

# Run the app
python -m app.main
```

### Debug Mode

Start with MongoDB Express UI at http://localhost:8081:
```bash
docker-compose --profile debug up -d
```

## License

MIT
