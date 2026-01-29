"""
API routes for the Interview Simulator.
"""

from app.services.code_executor_client import CodeExecutorClient
from fastapi import APIRouter, HTTPException, Depends

from app.db.mongodb import MongoDB, get_db
from app.models.requests import (
    CreateSessionRequest,
    MessageInfo,
    SendMessageRequest,
    SessionResponse,
    DisplayMessagesInfo,
    MessageResponse,
)
from app.models.session import InterviewInitInfo
from app.services.interview_service import InterviewService


router = APIRouter(prefix="/api/v1", tags=["interview"])


async def get_interview_service(db: MongoDB = Depends(get_db)) -> InterviewService:
    """Dependency for getting the interview service."""
    return InterviewService(db)


@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    service: InterviewService = Depends(get_interview_service)
):
    """
    Create a new interview session.
    
    The session will be initialized with:
    - The provided interview parameters (vacancy, stack, level, language)
    - An automatically generated question plan based on these parameters
    
    Returns the session ID and initial state.
    """
    init_info = InterviewInitInfo(
        vacancy=request.vacancy,
        description=request.description,
        stack=request.stack,
        level=request.level,
        language=request.language
    )
    
    try:
        session_id, session = await service.create_session(init_info)
        
        return SessionResponse(
            session_id=session_id,
            phase=session.phase,
            created_at=session.created_at,
            is_active=session.is_active,
            exchange_count=session.exchange_count,
            total_tokens_used=session.total_tokens_used
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    service: InterviewService = Depends(get_interview_service)
):
    """
    Get the current state of an interview session.
    
    Returns full session information including messages and init_info,
    allowing the client to resume the session.
    """
    session = await service.get_session(session_id)
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Convert display_messages to MessageInfo for client display (full history, never truncated)
    display_messages = DisplayMessagesInfo(
        interview=[
            MessageInfo(
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp
            )
            for msg in session.display_messages.interview
        ],
        live_coding=[
            MessageInfo(
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp
            )
            for msg in session.display_messages.live_coding
        ],
        final=[
            MessageInfo(
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp
            )
            for msg in session.display_messages.final
        ],
    )
    
    return SessionResponse(
        session_id=session_id,
        phase=session.phase,
        created_at=session.created_at,
        is_active=session.is_active,
        exchange_count=session.exchange_count,
        total_tokens_used=session.total_tokens_used,
        init_info=session.init_info,
        display_messages=display_messages,
        live_coding=session.live_coding
    )


@router.post("/sessions/{session_id}/start", response_model=MessageResponse)
async def start_interview(
    session_id: str,
    service: InterviewService = Depends(get_interview_service)
):
    """
    Start the interview session.
    
    This will have the AI interviewer introduce themselves and ask the first question.
    Returns a single JSON response with the interviewer's introduction.
    """
    session = await service.get_session(session_id)
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.messages:
        raise HTTPException(status_code=400, detail="Interview already started")
    
    try:
        return await service.start_interview(session_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/live_coding/start", response_model=MessageResponse)
async def start_live_coding(
    session_id: str,
    service: InterviewService = Depends(get_interview_service)
):
    """
    Start the live coding session.

    This will have the AI create the first coding challenge and introduce it.
    Returns a single JSON response with the live coding introduction.
    """
    session = await service.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.phase != "live_coding":
        raise HTTPException(status_code=400, detail="Session is not in live coding phase")

    try:
        return await service.start_live_coding(session_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/final/start", response_model=MessageResponse)
async def start_final_summary(
    session_id: str,
    service: InterviewService = Depends(get_interview_service)
):
    """
    Generate the final summary for a completed interview.
    """
    session = await service.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.phase != "final":
        raise HTTPException(status_code=400, detail="Session is not in final phase")

    try:
        return await service.start_final_summary(session_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/message", response_model=MessageResponse)
async def send_message(
    session_id: str,
    request: SendMessageRequest,
    service: InterviewService = Depends(get_interview_service)
):
    """
    Send a message in the interview session.
    
    The AI interviewer will:
    1. Process the candidate's response
    2. Update internal notes about the candidate
    3. Potentially adjust the question plan
    4. Ask the next question or follow-up
    
    Returns a single JSON response with the interviewer's reply.
    """
    session = await service.get_session(session_id)
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session is no longer active")
    
    try:
        return await service.process_message(session_id, request.message, request.current_code)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@router.get("/environments")
async def get_environments(
):
    service = CodeExecutorClient()
    print(f"Code Executor Base URL: {service._base_url}")
    """Get available execution environments."""
    return await service.list_environments()