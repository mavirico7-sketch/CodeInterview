"""
Session models for the Interview Simulator.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel, Field, model_validator


class InterviewPhase(str, Enum):
    """Current phase of the interview."""
    INTERVIEW = "interview"
    LIVE_CODING = "live_coding"
    FINAL = "final"


class CandidateLevel(str, Enum):
    """Candidate experience level."""
    INTERN = "intern"
    JUNIOR = "junior"
    MIDDLE = "middle"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"


class SessionMessage(BaseModel):
    """A single message in the interview conversation (user/assistant only)."""
    role: str  # "user", "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    token_count: int = 0


class TopicHistoryItem(BaseModel):
    """A completed topic/theme in the interview."""
    name: str  # Topic name (e.g., "Python basics", "SQL optimization")
    exchanges: int  # How many exchanges were spent on this topic


class InterviewProgress(BaseModel):
    """
    Tracks interview progress via topics and message exchanges.
    
    Exchange = one pair of (user message + assistant response)
    This is more natural than counting "questions" since one topic
    can involve multiple follow-up questions and discussions.
    """
    current_topic: Optional[str] = None  # Topic being discussed now
    current_topic_exchanges: int = 0  # Exchanges spent on current topic
    topics_history: list[TopicHistoryItem] = Field(default_factory=list)  # Completed topics


class InterviewInitInfo(BaseModel):
    """Initial interview configuration provided by the client."""
    vacancy: str  # e.g., "Backend Developer", "Data Scientist"
    stack: str  # e.g., "Python, FastAPI, PostgreSQL, Redis"
    level: CandidateLevel
    language: str  # e.g., "Russian", "English", "Spanish"


class Environment(BaseModel):
    """Execution environment for live coding."""
    id: str
    name: str
    description: str
    file_extension: str


class ExecutionResult(BaseModel):
    """Result of code execution."""
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    status: str


class CodeState(BaseModel):
    """Latest code and execution result."""
    code: str = ""
    execution_result: Optional[ExecutionResult] = None
    execution_result_shared: bool = False


class Challenge(BaseModel):
    """Current live coding challenge."""
    topic: str
    description: str
    initial_code: Optional[str] = None


class ChallengeHistoryItem(BaseModel):
    """Completed live coding challenge record."""
    topic: str
    description: str
    final_code: str


class LiveCodingState(BaseModel):
    """Live coding phase state."""
    environment: Optional[Environment] = None
    code_state: CodeState = Field(default_factory=CodeState)
    current_challenge: Optional[Challenge] = None
    challenges_history: list[ChallengeHistoryItem] = Field(default_factory=list)
    available_environments: list[Environment] = Field(default_factory=list)


class InterviewState(BaseModel):
    """Interview phase state."""
    progress: InterviewProgress = Field(default_factory=InterviewProgress)


class DisplayMessages(BaseModel):
    """Messages for client display, separated by phase."""
    interview: list[SessionMessage] = Field(default_factory=list)
    live_coding: list[SessionMessage] = Field(default_factory=list)
    final: list[SessionMessage] = Field(default_factory=list)


class InterviewSession(BaseModel):
    """Complete interview session document stored in MongoDB."""
    # MongoDB _id will be used as session_id
    init_info: InterviewInitInfo
    phase: InterviewPhase = InterviewPhase.INTERVIEW
    
    # Conversation messages (user and assistant only, no tool calls)
    # These may be truncated for LLM context management
    messages: list[SessionMessage] = Field(default_factory=list)
    # Full message history for client display (never truncated)
    display_messages: DisplayMessages = Field(default_factory=DisplayMessages)
    # Summarized context when messages are compressed
    context_summary: Optional[str] = None
    
    # Internal state (in English) - stored separately, not in messages
    # Candidate notes as structured list of entries
    candidate_notes: list[str] = Field(default_factory=list)
    # Interview progress tracking (topics, not individual questions)
    interview: InterviewState = Field(default_factory=InterviewState)
    # Live coding state (challenges, environment, code)
    live_coding: LiveCodingState = Field(default_factory=LiveCodingState)
    
    # Token tracking
    total_tokens_used: int = 0
    current_context_tokens: int = 0
    # Exchange count: one exchange = user message + assistant response
    exchange_count: int = 0
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Flags
    is_active: bool = True
    phase_completed_reason: Optional[str] = None
    final_summary: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        phase = data.get("phase")
        if phase == "questions":
            data["phase"] = InterviewPhase.INTERVIEW
        elif phase == "completed":
            data["phase"] = InterviewPhase.FINAL
        # Migrate legacy display_messages list
        display_messages = data.get("display_messages")
        if isinstance(display_messages, list):
            data["display_messages"] = {
                "interview": display_messages,
                "live_coding": [],
                "final": []
            }
        # Migrate legacy progress field
        if "progress" in data and "interview" not in data:
            data["interview"] = {"progress": data["progress"]}
        # Migrate legacy candidate notes from string to list
        candidate_notes = data.get("candidate_notes")
        if isinstance(candidate_notes, str):
            notes = candidate_notes.strip()
            data["candidate_notes"] = [notes] if notes else []
        return data

    class Config:
        use_enum_values = True
