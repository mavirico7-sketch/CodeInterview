"""
Interview Service - orchestrates the interview flow.
"""

import json
import logging
import re
from typing import Optional, Callable, Awaitable

from app.config import get_settings
from app.db.mongodb import MongoDB
from app.models.session import (
    InterviewSession,
    InterviewInitInfo,
    InterviewPhase,
    SessionMessage,
    InterviewProgress,
    ChallengeHistoryItem,
    Challenge,
    Environment,
)
from app.models.requests import MessageResponse
from app.services.llm_service import LLMService
from app.services.code_executor_client import CodeExecutorClient


class ToolResult:
    """Result of a tool execution."""
    def __init__(
        self, 
        message: str, 
        phase_changed: bool = False,
        new_phase: Optional[InterviewPhase] = None,
        should_wait_for_user: bool = False
    ):
        self.message = message
        self.phase_changed = phase_changed
        self.new_phase = new_phase
        self.should_wait_for_user = should_wait_for_user  # Stop generation, wait for user input


logger = logging.getLogger(__name__)


def _clean_llm_response(content: str) -> str:
    """
    Remove internal reasoning/chain-of-thought that some models leak into responses.

    Some models (especially through proxies) include their internal reasoning
    in the response content. This function strips obvious patterns.
    """
    if not content:
        return content

    original_content = content  # Save for debugging
    original_len = len(content)

    # Pattern 1: Lines starting with "A:" or "Assistant:" (internal monologue)
    content = re.sub(r'^A:\s*.+$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^Assistant:\s*.+$', '', content, flags=re.MULTILINE)

    # Pattern 2: Internal planning blocks (rules, reminders, meta-commentary)
    internal_patterns = [
        # Russian patterns
        r'^Правила:.*$',
        r'^Напоминание:.*$',
        r'^План:.*$',
        r'^После ответа кандидата.*$',
        r'^Когда он ответит.*$',
        r'^Не описывать новый challenge.*$',
        r'^Сначала кандидат должен.*$',
        r'^Это первый\.$',
        r'^Это \d+(-й|-ый)? challenge.*$',
        r'^max \d+ challenges.*$',
        r'^Exchanges completed:.*$',
        r'^Это \d+ exchanges.*$',
        r'^Жду (его |)ответ.*$',
        r'^Буду ждать.*$',
        r'^Теперь (я |)(должен|буду|жду).*$',
        # English patterns
        r'^Rules:.*$',
        r'^Reminder:.*$',
        r'^Plan:.*$',
        r'^Note to self:.*$',
        r'^Internal:.*$',
        r'^TODO:.*$',
        r'^Next steps?:.*$',
        r'^I (will|should|need to|must).*$',
        r'^Now I (will|should|need to|must).*$',
        r'^After (the |)candidate.*$',
        r'^When (the |they |he |she ).*respond.*$',
        r'^Waiting for.*$',
        r'^This is (the |)challenge \d+.*$',
        r'^Challenge \d+ of \d+.*$',
        r'^Max(imum)? \d+ challenges.*$',
        # Tool/instruction echoing
        r'^(Use |Using |I\'ll use )?(change_challenge|edit_code|change_phase|add_candidate_note).*$',
        r'^Do NOT.*$',
        r'^IMPORTANT:.*$',
        r'^Remember:.*$',
    ]
    for pattern in internal_patterns:
        content = re.sub(pattern, '', content, flags=re.MULTILINE | re.IGNORECASE)

    # Pattern 3: Remove blocks that look like internal reasoning (bracketed or prefixed)
    content = re.sub(r'\[Internal[^\]]*\]', '', content, flags=re.IGNORECASE)
    content = re.sub(r'###\s*Internal.*?###', '', content, flags=re.IGNORECASE | re.DOTALL)

    # Pattern 4: Remove excessive blank lines (more than 2 consecutive)
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Pattern 5: Trim whitespace
    content = content.strip()

    if len(content) < original_len:
        removed_chars = original_len - len(content)
        logger.info(
            "Cleaned LLM response: removed %d chars of internal reasoning",
            removed_chars
        )
        # Log raw content for debugging if significant amount was removed
        if removed_chars > 500:
            logger.warning(
                "Large amount of internal reasoning detected. First 1000 chars of RAW content: %s",
                original_content[:1000].replace("\n", "\\n")
            )

    return content


class InterviewService:
    """Service for managing interview sessions."""
    
    def __init__(self, db: MongoDB):
        self._db = db
        self._llm = LLMService()
        self._settings = get_settings()
        self._code_executor = CodeExecutorClient()
        self._max_tool_rounds = 3
        
        # Register tool handlers
        self._tool_handlers: dict[
            str,
            Callable[[InterviewSession, dict], tuple[InterviewSession, ToolResult]] | Callable[[InterviewSession, dict], Awaitable[tuple[InterviewSession, ToolResult]]]
        ] = {
            "add_candidate_note": self._handle_add_candidate_note,
            "delete_candidate_note": self._handle_delete_candidate_note,
            "edit_candidate_note": self._handle_edit_candidate_note,
            "change_phase": self._handle_change_phase,
            "change_challenge": self._handle_change_challenge,
            "edit_code": self._handle_edit_code,
        }

    async def _retry_empty_response(
        self,
        session: InterviewSession,
        initial_instruction: Optional[str],
        extra_messages: list[dict] | None,
        reason: str,
        session_id: Optional[str]
    ) -> dict:
        """Retry once with a system warning if the model returns empty content."""
        warning = {
            "role": "system",
            "content": (
                "WARNING: Your previous response was empty. "
                "You MUST respond with a non-empty candidate-facing message. "
                "Do not call tools."
            )
        }
        retry_messages = list(extra_messages or [])
        retry_messages.append(warning)
        logger.warning(
            "LLM empty response retry session=%s phase=%s reason=%s",
            session_id,
            session.phase,
            reason
        )
        return await self._llm.create_response(
            session,
            include_tools=False,
            initial_instruction=initial_instruction,
            extra_messages=retry_messages
        )

    async def _generate_final_summary(
        self,
        session: InterviewSession,
        session_id: Optional[str]
    ) -> tuple[str, dict]:
        """Generate the final summary after entering the final phase."""
        try:
            response = await self._llm.create_response(
                session,
                include_tools=False,
                initial_instruction=None,
                extra_messages=None
            )
        except Exception as e:
            logger.exception(
                "LLM final summary error session=%s phase=%s",
                session_id,
                session.phase
            )
            raise

        content = response.get("content", "")
        usage = response.get("usage", {}) or {}

        if not content:
            retry_response = await self._retry_empty_response(
                session,
                initial_instruction=None,
                extra_messages=[],
                reason="final_summary",
                session_id=session_id
            )
            content = retry_response.get("content", "")
            retry_usage = retry_response.get("usage", {}) or {}
            if retry_usage:
                for key, value in retry_usage.items():
                    usage[key] = usage.get(key, 0) + value

        # Clean any internal reasoning that leaked into the response
        content = _clean_llm_response(content)

        summary_preview = (content or "")[:500].replace("\n", "\\n")
        logger.info(
            "Final summary generated session=%s content_len=%s preview=%s",
            session_id,
            len(content or ""),
            summary_preview
        )

        return content, usage

    def _append_phase_marker(self, session: InterviewSession) -> None:
        """Add a system message marking the start of the current phase."""
        marker = f"[PHASE START] {session.phase}"
        for msg in reversed(session.messages):
            if msg.role == "system" and msg.content.startswith("[PHASE START]"):
                if msg.content == marker:
                    return
                break
        session.messages.append(SessionMessage(
            role="system",
            content=marker,
            token_count=self._llm.count_tokens(marker)
        ))

    def _record_final_summary(
        self,
        session: InterviewSession,
        summary: str
    ) -> InterviewSession:
        if not summary:
            return session
        session.final_summary = summary
        assistant_msg = SessionMessage(
            role="assistant",
            content=summary,
            token_count=self._llm.count_tokens(summary)
        )
        session.messages.append(assistant_msg)
        session.display_messages.final.append(assistant_msg)
        return session
    
    # ========== Tool Handlers ==========
    
    def _handle_add_candidate_note(
        self, 
        session: InterviewSession, 
        arguments: dict
    ) -> tuple[InterviewSession, ToolResult]:
        """
        Handle add_candidate_note tool call.
        Appends a new entry to candidate notes.
        """
        entry = (arguments.get("entry") or "").strip()
        if not entry:
            return session, ToolResult("No entry provided.")
        session.candidate_notes.append(entry)
        return session, ToolResult(f"Note added. Total notes: {len(session.candidate_notes)}.")

    def _handle_delete_candidate_note(
        self,
        session: InterviewSession,
        arguments: dict
    ) -> tuple[InterviewSession, ToolResult]:
        """
        Handle delete_candidate_note tool call.
        Removes an entry by 1-based index.
        """
        index = arguments.get("index")
        try:
            idx = int(index)
        except (TypeError, ValueError):
            return session, ToolResult("Invalid index.")
        if idx < 1 or idx > len(session.candidate_notes):
            return session, ToolResult("Index out of range.")
        removed = session.candidate_notes.pop(idx - 1)
        return session, ToolResult(f"Deleted note {idx}: {removed}")

    def _handle_edit_candidate_note(
        self,
        session: InterviewSession,
        arguments: dict
    ) -> tuple[InterviewSession, ToolResult]:
        """
        Handle edit_candidate_note tool call.
        Replaces an entry by 1-based index.
        """
        index = arguments.get("index")
        entry = (arguments.get("entry") or "").strip()
        try:
            idx = int(index)
        except (TypeError, ValueError):
            return session, ToolResult("Invalid index.")
        if idx < 1 or idx > len(session.candidate_notes):
            return session, ToolResult("Index out of range.")
        if not entry:
            return session, ToolResult("No entry provided.")
        session.candidate_notes[idx - 1] = entry
        return session, ToolResult(f"Updated note {idx}.")
    
    
    async def _handle_change_phase(
        self,
        session: InterviewSession,
        arguments: dict
    ) -> tuple[InterviewSession, ToolResult]:
        """
        Handle change_phase tool call.
        """
        target_phase = arguments.get("phase")
        if target_phase not in [InterviewPhase.LIVE_CODING, InterviewPhase.FINAL]:
            return session, ToolResult("Invalid phase transition requested.")

        if session.phase == InterviewPhase.INTERVIEW and target_phase != InterviewPhase.LIVE_CODING:
            return session, ToolResult("Cannot transition to that phase from interview.")

        if session.phase == InterviewPhase.LIVE_CODING and target_phase != InterviewPhase.FINAL:
            return session, ToolResult("Cannot transition to that phase from live coding.")

        if session.phase == InterviewPhase.FINAL:
            return session, ToolResult("Already in final phase.")

        if target_phase == InterviewPhase.LIVE_CODING:
            await self._ensure_available_environments(session)
            environment_data = arguments.get("environment")
            if environment_data:
                session.live_coding.environment = Environment(**environment_data)
            else:
                session.live_coding.environment = self._select_environment(session)
            session.phase = InterviewPhase.LIVE_CODING
            return session, ToolResult(
                "Transitioned to live coding phase.",
                phase_changed=True,
                new_phase=InterviewPhase.LIVE_CODING
            )

        session.phase = InterviewPhase.FINAL
        session.is_active = False
        session.final_summary = arguments.get("final_summary")
        return session, ToolResult(
            "Transitioned to final phase.",
            phase_changed=True,
            new_phase=InterviewPhase.FINAL
        )

    def _handle_change_challenge(
        self,
        session: InterviewSession,
        arguments: dict
    ) -> tuple[InterviewSession, ToolResult]:
        """Handle change_challenge tool call."""
        if session.phase != InterviewPhase.LIVE_CODING:
            return session, ToolResult("change_challenge is only valid in live coding phase.")

        max_challenges = self._settings.live_coding.max_challenges
        total_seen = len(session.live_coding.challenges_history)
        if session.live_coding.current_challenge:
            total_seen += 1
        if total_seen >= max_challenges:
            return session, ToolResult("Maximum challenges reached. Use change_phase to finish.")

        if session.live_coding.current_challenge:
            session.live_coding.challenges_history.append(
                ChallengeHistoryItem(
                    topic=session.live_coding.current_challenge.topic,
                    description=session.live_coding.current_challenge.description,
                    final_code=session.live_coding.code_state.code
                )
            )

        topic = arguments.get("topic", "General coding challenge")
        description = arguments.get("description", "")
        initial_code = arguments.get("initial_code")
        session.live_coding.current_challenge = Challenge(
            topic=topic,
            description=description,
            initial_code=initial_code
        )

        if initial_code is not None:
            session.live_coding.code_state.code = initial_code
            session.live_coding.code_state.execution_result = None
            session.live_coding.code_state.execution_result_shared = False

        return session, ToolResult(f"Challenge set: '{topic}'.")

    def _handle_edit_code(
        self,
        session: InterviewSession,
        arguments: dict
    ) -> tuple[InterviewSession, ToolResult]:
        """Handle edit_code tool call."""
        if session.phase != InterviewPhase.LIVE_CODING:
            return session, ToolResult("edit_code is only valid in live coding phase.")

        new_code = arguments.get("new_code")
        if new_code is None:
            return session, ToolResult("No code provided for edit.")

        if new_code != session.live_coding.code_state.code:
            session.live_coding.code_state.code = new_code
            session.live_coding.code_state.execution_result = None
            session.live_coding.code_state.execution_result_shared = False
            return session, ToolResult("Code updated.")

        return session, ToolResult("Code unchanged.")

    
    # ========== Session Management ==========
    
    async def create_session(self, init_info: InterviewInitInfo) -> tuple[str, InterviewSession]:
        """Create a new interview session."""
        session = InterviewSession(init_info=init_info)
        
        # Session starts with empty progress - model chooses first topic on start
        session.interview.progress = InterviewProgress()
        
        # Save to database
        session_id = await self._db.create_session(session)
        
        return session_id, session
    
    async def get_session(self, session_id: str) -> Optional[InterviewSession]:
        """Get a session by ID."""
        return await self._db.get_session(session_id)
    
    # ========== Tool Processing ==========
    
    async def _process_tool_call(
        self, 
        session: InterviewSession, 
        tool_name: str, 
        arguments: dict
    ) -> tuple[InterviewSession, ToolResult]:
        """
        Process a single tool call using registered handlers.
        """
        handler = self._tool_handlers.get(tool_name)
        
        if handler is None:
            return session, ToolResult(f"Unknown tool: {tool_name}")
        
        result = handler(session, arguments)
        if isinstance(result, tuple):
            return result
        return await result
    
    def _parse_tool_arguments(self, arguments_str: str) -> dict:
        """Parse tool arguments from JSON string."""
        if not arguments_str:
            return {}
        try:
            return json.loads(arguments_str)
        except json.JSONDecodeError:
            return {}

    def _format_tool_result_content(
        self,
        tool_name: str,
        result: ToolResult,
        session: InterviewSession
    ) -> str:
        """Format tool results for LLM tool messages."""
        if tool_name == "change_challenge" and session.live_coding.current_challenge:
            # Add challenge details to help model understand the change
            challenge = session.live_coding.current_challenge
            details = f"\nNew challenge topic: {challenge.topic}\nDescription: {challenge.description}"
            if challenge.initial_code:
                details += f"\nInitial code has been set in the editor."
            return result.message + details
        return result.message

    async def _run_agent_loop(
        self,
        session: InterviewSession,
        initial_instruction: Optional[str] = None,
        stop_on_phase_change: bool = False,
        session_id: Optional[str] = None
    ) -> tuple[str, dict, bool, Optional[InterviewPhase], Optional[str]]:
        """
        Run the LLM with tools in a loop until a candidate-facing response is produced.
        Returns (final_content, usage, phase_changed, new_phase, error).
        """
        tool_context_messages: list[dict] = []
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        phase_changed = False
        new_phase = None
        last_content = ""

        for round_index in range(self._max_tool_rounds):
            try:
                response = await self._llm.create_response(
                    session,
                    include_tools=True,
                    initial_instruction=initial_instruction,
                    extra_messages=tool_context_messages
                )
            except Exception as e:
                logger.exception(
                    "LLM tool round error session=%s phase=%s round=%s",
                    session_id,
                    session.phase,
                    round_index
                )
                return "", usage_total, phase_changed, new_phase, str(e)

            collected_content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])
            usage = response.get("usage", {})

            last_content = collected_content

            if usage:
                for key in usage_total:
                    usage_total[key] += usage.get(key, 0)

            if collected_content or tool_calls:
                content_preview = (collected_content or "")[:500].replace("\n", "\\n")
                logger.info(
                    "LLM round output session=%s phase=%s round=%s content_len=%s tool_calls=%s preview=%s",
                    session_id,
                    session.phase,
                    round_index,
                    len(collected_content or ""),
                    len(tool_calls),
                    content_preview
                )

            # No tool calls — this is the final candidate-facing response
            if not tool_calls:
                if not collected_content:
                    retry_response = await self._retry_empty_response(
                        session,
                        initial_instruction,
                        tool_context_messages,
                        reason="no_tool_calls",
                        session_id=session_id
                    )
                    collected_content = retry_response.get("content", "")
                    retry_usage = retry_response.get("usage", {})
                    if retry_usage:
                        for key in usage_total:
                            usage_total[key] += retry_usage.get(key, 0)
                return _clean_llm_response(collected_content), usage_total, phase_changed, new_phase, None

            # Model called tools — preserve content in assistant message
            assistant_tool_msg = {
                "role": "assistant",
                "content": collected_content or "",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"]
                        }
                    }
                    for tc in tool_calls
                ]
            }
            tool_context_messages.append(assistant_tool_msg)

            # Process each tool call
            for tc in tool_calls:
                args = self._parse_tool_arguments(tc["arguments"])
                try:
                    session, result = await self._process_tool_call(session, tc["name"], args)
                except Exception as e:
                    logger.exception(
                        "Tool call exception session=%s phase=%s tool=%s args=%s",
                        session_id,
                        session.phase,
                        tc["name"],
                        tc["arguments"]
                    )
                    result = ToolResult(f"Tool execution failed: {str(e)}")
                logger.info(
                    "Tool result session=%s phase=%s tool=%s result=%s",
                    session_id,
                    session.phase,
                    tc["name"],
                    result.message
                )
                if result.phase_changed:
                    phase_changed = True
                    new_phase = result.new_phase

                tool_context_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": self._format_tool_result_content(tc["name"], result, session)
                })

            if stop_on_phase_change and phase_changed:
                logger.info(
                    "LLM tool round stopped session=%s phase=%s round=%s",
                    session_id,
                    session.phase,
                    round_index
                )
                return "", usage_total, phase_changed, new_phase, None

            # Loop continues — next iteration calls LLM with tool results in context

        return _clean_llm_response(last_content), usage_total, phase_changed, new_phase, None
    
    # ========== Context Management ==========
    
    async def _check_and_summarize_context(self, session: InterviewSession) -> InterviewSession:
        """
        Check if context exceeds limit and summarize chat messages if needed.
        Only summarizes conversation, not candidate notes or question plan.
        """
        settings = self._settings
        
        # Calculate current context size (includes system prompt with notes/plan)
        messages_for_api = self._llm.build_messages(session)
        current_tokens = self._llm.count_messages_tokens(messages_for_api)
        session.current_context_tokens = current_tokens
        
        if current_tokens > settings.interview.context_token_limit:
            # Need to summarize chat messages
            preserve_count = settings.interview.preserve_initial_messages
            messages_to_keep_end = 4  # Keep last 2 exchanges
            
            if len(session.messages) > preserve_count + messages_to_keep_end:
                # Extract messages to summarize
                start_idx = preserve_count
                end_idx = len(session.messages) - messages_to_keep_end
                
                messages_to_summarize = [
                    {"role": msg.role, "content": msg.content}
                    for msg in session.messages[start_idx:end_idx]
                ]
                
                if messages_to_summarize:
                    # Generate summary
                    summary_result = await self._llm.summarize_context(messages_to_summarize)
                    
                    # Update session
                    if session.context_summary:
                        session.context_summary = f"{session.context_summary}\n\n[CONTINUED]\n{summary_result['summary']}"
                    else:
                        session.context_summary = summary_result["summary"]
                    
                    # Remove summarized messages
                    session.messages = (
                        session.messages[:preserve_count] + 
                        session.messages[end_idx:]
                    )
                    
                    session.total_tokens_used += summary_result["usage"]["total_tokens"]
        
        return session
    
    async def _check_token_limit(self, session: InterviewSession) -> bool:
        """Check if total token limit has been reached."""
        return session.total_tokens_used >= self._settings.interview.total_token_limit
    
    def _check_exchange_limit(self, session: InterviewSession) -> bool:
        """Check if maximum exchanges limit has been reached."""
        return session.exchange_count >= self._settings.interview.max_exchanges

    async def _ensure_available_environments(self, session: InterviewSession) -> None:
        if session.live_coding.available_environments:
            return
        try:
            session.live_coding.available_environments = await self._code_executor.list_environments()
        except Exception:
            session.live_coding.available_environments = []

    def _select_environment(self, session: InterviewSession) -> Optional[Environment]:
        """Select the best environment based on interview context."""
        environments = session.live_coding.available_environments or []
        if not environments:
            return None

        stack = f"{session.init_info.vacancy} {session.init_info.stack}".lower()
        ml_keywords = ["ml", "machine learning", "pytorch", "numpy", "pandas", "scikit", "sklearn"]

        if any(keyword in stack for keyword in ml_keywords):
            for env in environments:
                env_text = f"{env.name} {env.description}".lower()
                if "ml" in env_text or "numpy" in env_text or "scikit" in env_text:
                    return env

        for env in environments:
            if env.name.lower() == "python":
                return env

        return environments[0]
    
    # ========== Message Processing ==========
    
    async def process_message(
        self,
        session_id: str,
        user_message: str,
        current_code: Optional[str] = None
    ) -> MessageResponse:
        """
        Process a user message and return a full response.
        
        Multi-step LLM approach:
        - Model can call tools first
        - Tools are executed and results provided back to the model
        - Candidate-facing response is returned after tools (if any) are executed
        """
        session = await self.get_session(session_id)
        user_preview = user_message[:200].replace("\n", "\\n")
        logger.info(
            "process_message called session=%s phase=%s user_preview=%s",
            session_id,
            session.phase,
            user_preview
        )
        
        if session is None:
            raise LookupError("Session not found")
        
        if not session.is_active:
            raise ValueError("Session is no longer active")

        if session.phase == InterviewPhase.FINAL:
            raise ValueError("Interview is already completed.")

        if user_message.strip() == "/next_phase":
            response_phase = session.phase
            final_content = ""
            display_msg = SessionMessage(
                role="user",
                content=user_message,
                token_count=self._llm.count_tokens(user_message)
            )
            if response_phase == InterviewPhase.INTERVIEW:
                session.display_messages.interview.append(display_msg)
            elif response_phase == InterviewPhase.LIVE_CODING:
                session.display_messages.live_coding.append(display_msg)

            await self._ensure_available_environments(session)
            if response_phase == InterviewPhase.INTERVIEW:
                session, result = await self._handle_change_phase(session, {"phase": "live_coding"})
                debug_note = "(debug command used): switched to live_coding phase"
            else:
                session, result = await self._handle_change_phase(
                    session,
                    {"phase": "final", "final_summary": "Transitioned by /next_phase."}
                )
                debug_note = "(debug command used): switched to final phase"

            session.messages.append(SessionMessage(
                role="assistant",
                content=debug_note,
                token_count=self._llm.count_tokens(debug_note)
            ))

            await self._db.update_session(session_id, session)
            return MessageResponse(
                session_id=session_id,
                content=final_content or "",
                phase=session.phase,
                exchange_count=session.exchange_count,
                total_tokens_used=session.total_tokens_used,
                phase_changed=result.phase_changed,
                is_phase_complete=session.phase == InterviewPhase.FINAL
            )

        await self._ensure_available_environments(session)

        response_phase = session.phase
        message_content = user_message
        if session.phase == InterviewPhase.LIVE_CODING:
            code_changed = current_code is not None and current_code != session.live_coding.code_state.code
            should_share_exec = (
                session.live_coding.code_state.execution_result is not None
                and not session.live_coding.code_state.execution_result_shared
                and not code_changed
            )
            if code_changed:
                session.live_coding.code_state.code = current_code or ""
                session.live_coding.code_state.execution_result = None
                session.live_coding.code_state.execution_result_shared = False
            if code_changed or should_share_exec:
                message_content += f"\n\n--- CODE UPDATE ---\n{session.live_coding.code_state.code}"
            if should_share_exec:
                exec_result = session.live_coding.code_state.execution_result
                message_content += (
                    "\n\n--- EXECUTION RESULT ---\n"
                    f"stdout:\n{exec_result.stdout}\n"
                    f"stderr:\n{exec_result.stderr}\n"
                    f"exit_code: {exec_result.exit_code}\n"
                    f"execution_time: {exec_result.execution_time}\n"
                    f"status: {exec_result.status}"
                )
                session.live_coding.code_state.execution_result_shared = True

        user_msg = SessionMessage(
            role="user",
            content=message_content,
            token_count=self._llm.count_tokens(message_content)
        )
        session.messages.append(user_msg)

        display_msg = SessionMessage(
            role="user",
            content=user_message,
            token_count=self._llm.count_tokens(user_message)
        )
        if session.phase == InterviewPhase.INTERVIEW:
            session.display_messages.interview.append(display_msg)
        elif session.phase == InterviewPhase.LIVE_CODING:
            session.display_messages.live_coding.append(display_msg)

        # Save user message immediately to prevent data loss if LLM call fails
        await self._db.update_session(session_id, session)

        collected_content = ""
        phase_changed = False
        new_phase = None
        llm_error = None

        try:
            session = await self._check_and_summarize_context(session)

            logger.info("Calling LLM session=%s phase=%s", session_id, session.phase)
            stop_on_phase_change = True
            collected_content, usage, phase_changed, new_phase, error = await self._run_agent_loop(
                session,
                stop_on_phase_change=stop_on_phase_change,
                session_id=session_id
            )
            if error:
                logger.error(
                    "LLM agent loop error session=%s phase=%s error=%s",
                    session_id,
                    session.phase,
                    error
                )
                llm_error = error
            if usage:
                session.total_tokens_used += usage.get("total_tokens", 0)

            if phase_changed and new_phase == InterviewPhase.FINAL and not collected_content:
                collected_content, summary_usage = await self._generate_final_summary(
                    session,
                    session_id=session_id
                )
                if summary_usage:
                    session.total_tokens_used += summary_usage.get("total_tokens", 0)

            # Fallback for LIVE_CODING phase
            if (
                not collected_content
                and response_phase == InterviewPhase.LIVE_CODING
                and not (phase_changed and new_phase == InterviewPhase.FINAL)
            ):
                challenge = session.live_coding.current_challenge
                if challenge:
                    collected_content = (
                        f"New challenge: {challenge.topic}\n\n"
                        f"{challenge.description}\n\n"
                        "Start coding in the editor and let me know when you're ready to discuss."
                    )

            # Fallback for INTERVIEW phase
            if (
                not collected_content
                and response_phase == InterviewPhase.INTERVIEW
                and not phase_changed
            ):
                collected_content = (
                    "I apologize, but I couldn't generate a response. "
                    "Could you please repeat your answer or rephrase it?"
                )
                logger.warning(
                    "Using fallback response for empty content session=%s phase=%s",
                    session_id,
                    session.phase
                )

        except Exception as e:
            logger.exception(
                "Unexpected error in LLM processing session=%s phase=%s",
                session_id,
                session.phase
            )
            llm_error = str(e)
            if not collected_content:
                collected_content = (
                    "I apologize, but I encountered a technical issue. "
                    "Please try sending your message again."
                )

        final_preview = (collected_content or "")[:500].replace("\n", "\\n")
        logger.info(
            "LLM final response session=%s phase=%s content_len=%s error=%s preview=%s",
            session_id,
            session.phase,
            len(collected_content or ""),
            llm_error,
            final_preview
        )

        final_content = collected_content
        if final_content:
            if phase_changed and new_phase == InterviewPhase.FINAL:
                session = self._record_final_summary(session, final_content)
            else:
                assistant_msg = SessionMessage(
                    role="assistant",
                    content=final_content,
                    token_count=self._llm.count_tokens(final_content)
                )
                session.messages.append(assistant_msg)

                if response_phase == InterviewPhase.INTERVIEW:
                    session.display_messages.interview.append(assistant_msg)
                elif response_phase == InterviewPhase.LIVE_CODING:
                    session.display_messages.live_coding.append(assistant_msg)

        if final_content and not llm_error:
            session.exchange_count += 1

        # Always save session to preserve state
        save_success = await self._db.update_session(session_id, session)
        if not save_success:
            logger.error(
                "Failed to save session session=%s phase=%s",
                session_id,
                session.phase
            )

        return MessageResponse(
            session_id=session_id,
            content=final_content or "",
            phase=session.phase,
            exchange_count=session.exchange_count,
            total_tokens_used=session.total_tokens_used,
            phase_changed=phase_changed,
            is_phase_complete=session.phase == InterviewPhase.FINAL
        )
    
    async def start_interview(self, session_id: str) -> MessageResponse:
        """
        Start the interview by getting the first message from the interviewer.
        
        Multi-step LLM call - model can call tools first, then respond with final message.
        """
        session = await self.get_session(session_id)
        
        if session is None:
            raise LookupError("Session not found")
        
        if session.messages:
            raise ValueError("Interview already started")

        if session.phase != InterviewPhase.INTERVIEW:
            raise ValueError("Interview is not in interview phase")

        await self._ensure_available_environments(session)

        if not session.messages:
            self._append_phase_marker(session)
        
        # Initial instruction (not stored in messages)
        initial_instruction = "[Interview started. Introduce yourself briefly and ask your first question.]"
        
        collected_content, usage, phase_changed, _, error = await self._run_agent_loop(
            session,
            initial_instruction=initial_instruction,
            session_id=session_id
        )
        if error:
            raise RuntimeError(error)
        if usage:
            session.total_tokens_used += usage.get("total_tokens", 0)

        start_preview = (collected_content or "")[:500].replace("\n", "\\n")
        logger.info(
            "START interview response session=%s content_len=%s preview=%s",
            session_id,
            len(collected_content or ""),
            start_preview
        )

        final_content = collected_content
        if final_content:
            assistant_msg = SessionMessage(
                role="assistant",
                content=final_content,
                token_count=self._llm.count_tokens(final_content)
            )
            session.messages.append(assistant_msg)
            session.display_messages.interview.append(assistant_msg)
        
        # Save session
        await self._db.update_session(session_id, session)
        return MessageResponse(
            session_id=session_id,
            content=final_content or "",
            phase=session.phase,
            exchange_count=session.exchange_count,
            total_tokens_used=session.total_tokens_used,
            phase_changed=phase_changed,
            is_phase_complete=session.phase == InterviewPhase.FINAL
        )

    async def start_live_coding(self, session_id: str) -> MessageResponse:
        """
        Start live coding by creating the first challenge and prompting the candidate.
        """
        session = await self.get_session(session_id)

        if session is None:
            raise LookupError("Session not found")

        if session.phase != InterviewPhase.LIVE_CODING:
            raise ValueError("Live coding is not active")

        if session.display_messages.live_coding:
            raise ValueError("Live coding already started")

        await self._ensure_available_environments(session)
        self._append_phase_marker(session)

        initial_instruction = (
            "[Live coding started. Create the first challenge using change_challenge, "
            "then greet the candidate and present the task. "
            "Do NOT include any internal reasoning or planning in your response.]"
        )

        collected_content, usage, phase_changed, _, error = await self._run_agent_loop(
            session,
            initial_instruction=initial_instruction,
            session_id=session_id
        )
        if error:
            raise RuntimeError(error)
        if usage:
            session.total_tokens_used += usage.get("total_tokens", 0)

        start_preview = (collected_content or "")[:500].replace("\n", "\\n")
        logger.info(
            "START live_coding response session=%s content_len=%s preview=%s",
            session_id,
            len(collected_content or ""),
            start_preview
        )

        final_content = collected_content
        if final_content:
            assistant_msg = SessionMessage(
                role="assistant",
                content=final_content,
                token_count=self._llm.count_tokens(final_content)
            )
            session.messages.append(assistant_msg)
            session.display_messages.live_coding.append(assistant_msg)

        await self._db.update_session(session_id, session)
        return MessageResponse(
            session_id=session_id,
            content=final_content or "",
            phase=session.phase,
            exchange_count=session.exchange_count,
            total_tokens_used=session.total_tokens_used,
            phase_changed=phase_changed,
            is_phase_complete=session.phase == InterviewPhase.FINAL
        )

    async def start_final_summary(self, session_id: str) -> MessageResponse:
        """
        Generate the final summary if it has not been created yet.
        """
        session = await self.get_session(session_id)

        if session is None:
            raise LookupError("Session not found")

        if session.phase != InterviewPhase.FINAL:
            raise ValueError("Interview is not in final phase")

        if session.display_messages.final:
            raise ValueError("Final summary already generated")

        self._append_phase_marker(session)

        collected_content, summary_usage = await self._generate_final_summary(
            session,
            session_id=session_id
        )
        if summary_usage:
            session.total_tokens_used += summary_usage.get("total_tokens", 0)

        session = self._record_final_summary(session, collected_content)

        await self._db.update_session(session_id, session)
        return MessageResponse(
            session_id=session_id,
            content=collected_content or "",
            phase=session.phase,
            exchange_count=session.exchange_count,
            total_tokens_used=session.total_tokens_used,
            phase_changed=False,
            is_phase_complete=session.phase == InterviewPhase.FINAL
        )
