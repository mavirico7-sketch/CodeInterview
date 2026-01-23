"""
Request and Response models for API endpoints.
"""

from datetime import datetime
from typing import Optional, Any

from pydantic import BaseModel, Field, model_validator

from app.models.session import CandidateLevel, InterviewPhase, InterviewInitInfo, LiveCodingState


class CreateSessionRequest(BaseModel):
    """Request body for creating a new interview session."""
    vacancy: str = Field(..., description="Job position title", min_length=1)
    stack: str = Field(..., description="Technology stack (comma-separated)", min_length=1)
    level: CandidateLevel = Field(..., description="Candidate experience level")
    language: str = Field(default="English", description="Interview language for communication")


class SendMessageRequest(BaseModel):
    """Request body for sending a message in an interview session."""
    message: Optional[str] = Field(None, description="User's message content")
    content: Optional[str] = Field(None, description="Deprecated: use message")
    current_code: Optional[str] = Field(default=None, description="Current code from editor")

    @model_validator(mode="before")
    @classmethod
    def _normalize_message(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "message" not in data and "content" in data:
            data["message"] = data.get("content")
        return data

    @model_validator(mode="after")
    def _validate_message(self) -> "SendMessageRequest":
        if not self.message or not self.message.strip():
            raise ValueError("message is required")
        return self


class MessageInfo(BaseModel):
    """Message info for client display."""
    role: str
    content: str
    timestamp: datetime


class DisplayMessagesInfo(BaseModel):
    """Messages grouped by phase for client display."""
    interview: list[MessageInfo] = Field(default_factory=list)
    live_coding: list[MessageInfo] = Field(default_factory=list)
    final: list[MessageInfo] = Field(default_factory=list)


class SessionResponse(BaseModel):
    """Response with session information."""
    session_id: str
    phase: InterviewPhase
    created_at: datetime
    is_active: bool
    exchange_count: int  # Number of message exchanges (user + assistant pairs)
    total_tokens_used: int
    # Optional fields for session resumption
    init_info: Optional[InterviewInitInfo] = None
    # Full message history for client display (never truncated)
    display_messages: Optional[DisplayMessagesInfo] = None
    live_coding: Optional[LiveCodingState] = None


class MessageResponse(BaseModel):
    """Response for a non-streaming message."""
    session_id: str
    content: str
    phase: InterviewPhase
    exchange_count: int
    total_tokens_used: int
    phase_changed: bool = False
    is_phase_complete: bool = False


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None

