"""
LLM Service for OpenAI API interactions with tool calling.
"""

import json

import tiktoken
from openai import AsyncOpenAI

from app.config import get_settings, get_prompts
from app.models.session import (
    InterviewSession,
    InterviewProgress,
    InterviewPhase,
    LiveCodingState,
    Environment,
)


class LLMService:
    """Service for interacting with OpenAI API."""
    
    def __init__(self):
        settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url
        )
        self._model = settings.llm.model
        self._temperature = settings.llm.temperature
        self._max_tokens = settings.llm.max_response_tokens
        
        # Token counter
        try:
            self._encoding = tiktoken.encoding_for_model(self._model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self._encoding.encode(text))
    
    def count_messages_tokens(self, messages: list[dict]) -> int:
        """Count total tokens in a list of messages."""
        total = 0
        for msg in messages:
            total += 4  # message overhead
            for key, value in msg.items():
                if isinstance(value, str):
                    total += self.count_tokens(value)
                elif isinstance(value, list):
                    total += self.count_tokens(json.dumps(value))
        total += 2  # assistant priming
        return total

    def get_tools_definition(self, session: InterviewSession) -> list[dict]:
        """Get the tools definition for OpenAI API."""
        prompts = get_prompts()
        base_tools = [
            {
                "type": "function",
                "function": {
                    "name": "add_candidate_note",
                    "description": prompts.tools.add_candidate_note.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entry": {
                                "type": "string",
                                "description": "New note entry to append."
                            }
                        },
                        "required": ["entry"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_candidate_note",
                    "description": prompts.tools.delete_candidate_note.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "1-based index of the note to delete."
                            }
                        },
                        "required": ["index"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_candidate_note",
                    "description": prompts.tools.edit_candidate_note.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "1-based index of the note to edit."
                            },
                            "entry": {
                                "type": "string",
                                "description": "Replacement text for the note."
                            }
                        },
                        "required": ["index", "entry"]
                    }
                }
            }
        ]

        if session.phase == InterviewPhase.INTERVIEW:
            return base_tools + [
                {
                    "type": "function",
                    "function": {
                        "name": "change_phase",
                        "description": prompts.tools.change_phase.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "phase": {
                                    "type": "string",
                                    "enum": ["live_coding"]
                                },
                                "environment": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "name": {"type": "string"},
                                        "description": {"type": "string"},
                                        "file_extension": {"type": "string"}
                                    }
                                }
                            },
                            "required": ["phase"]
                        }
                    }
                }
            ]

        if session.phase == InterviewPhase.LIVE_CODING:
            return base_tools + [
                {
                    "type": "function",
                    "function": {
                        "name": "change_challenge",
                        "description": prompts.tools.change_challenge.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string"},
                                "description": {"type": "string"},
                                "initial_code": {"type": "string"}
                            },
                            "required": ["topic", "description"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "edit_code",
                        "description": prompts.tools.edit_code.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "new_code": {"type": "string"},
                                "explanation": {"type": "string"}
                            },
                            "required": ["new_code", "explanation"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "change_phase",
                        "description": prompts.tools.change_phase.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "phase": {
                                    "type": "string",
                                    "enum": ["final"]
                                },
                                "final_summary": {"type": "string"}
                            },
                            "required": ["phase"]
                        }
                    }
                }
            ]

        return []
    
    def _format_progress(self, progress: InterviewProgress) -> str:
        """Format interview progress for the system prompt."""
        lines = []
        
        # Topics already discussed
        if progress.topics_history:
            lines.append("**Topics Discussed:**")
            for i, topic in enumerate(progress.topics_history, 1):
                lines.append(f"  {i}. {topic.name} ({topic.exchanges} exchanges)")
        else:
            lines.append("**Topics Discussed:** None yet - this is the start of the interview")
        
        lines.append("")
        
        # Current topic
        if progress.current_topic:
            lines.append(f"**Current Topic:** {progress.current_topic}")
            lines.append(f"   Exchanges on this topic: {progress.current_topic_exchanges}")
        else:
            lines.append("**Current Topic:** None - choose your first topic to discuss")
        
        return "\n".join(lines)

    def _format_environments(self, environments: list[Environment]) -> str:
        if not environments:
            return "None available"
        lines = []
        for env in environments:
            lines.append(f"- {env.name} ({env.description}), file: {env.file_extension}, id: {env.id}")
        return "\n".join(lines)

    def _format_selected_environment(self, live_coding: LiveCodingState) -> str:
        if live_coding.environment is None:
            return "None selected"
        env = live_coding.environment
        return f"{env.name} ({env.description}), file: {env.file_extension}, id: {env.id}"

    def _format_current_challenge(self, live_coding: LiveCodingState) -> str:
        if live_coding.current_challenge is None:
            return "None - create one with change_challenge."
        challenge = live_coding.current_challenge
        initial = f"\nInitial code:\n{challenge.initial_code}" if challenge.initial_code else ""
        return f"Topic: {challenge.topic}\nDescription: {challenge.description}{initial}"

    def _format_challenges_history(self, live_coding: LiveCodingState) -> str:
        if not live_coding.challenges_history:
            return "None yet"
        lines = []
        for i, item in enumerate(live_coding.challenges_history, 1):
            lines.append(f"{i}. {item.topic} - {item.description}")
        return "\n".join(lines)

    def _format_candidate_notes(self, notes: list[str]) -> str:
        if not notes:
            return "No notes yet. Add after evaluating responses."
        lines = [f"{i}. {note}" for i, note in enumerate(notes, 1)]
        return "\n".join(lines)

    def _format_code_state(self, live_coding: LiveCodingState) -> str:
        if not live_coding.code_state.code:
            return "No code submitted yet."
        return "Latest code was provided in the most recent [CODE UPDATE] message."
    
    def build_system_prompt(self, session: InterviewSession) -> str:
        """Build the system prompt with session context."""
        prompts = get_prompts()
        settings = get_settings()
        
        # Format candidate notes
        notes_str = self._format_candidate_notes(session.candidate_notes)

        if session.phase == InterviewPhase.INTERVIEW:
            progress_str = self._format_progress(session.interview.progress)
            remaining_exchanges = settings.interview.max_exchanges - session.exchange_count
            available_envs = self._format_environments(session.live_coding.available_environments)
            return prompts.interview_system_prompt.format(
                vacancy=session.init_info.vacancy,
                stack=session.init_info.stack,
                level=session.init_info.level,
                language=session.init_info.language,
                exchange_count=session.exchange_count,
                max_exchanges=settings.interview.max_exchanges,
                remaining_exchanges=remaining_exchanges,
                topics_count=len(session.interview.progress.topics_history),
                progress=progress_str,
                candidate_notes=notes_str,
                available_environments=available_envs
            )

        if session.phase == InterviewPhase.LIVE_CODING:
            return prompts.live_coding_system_prompt.format(
                vacancy=session.init_info.vacancy,
                stack=session.init_info.stack,
                level=session.init_info.level,
                language=session.init_info.language,
                max_challenges=settings.live_coding.max_challenges,
                selected_environment=self._format_selected_environment(session.live_coding),
                available_environments=self._format_environments(session.live_coding.available_environments),
                current_challenge=self._format_current_challenge(session.live_coding),
                challenges_history=self._format_challenges_history(session.live_coding),
                code_state=self._format_code_state(session.live_coding),
                candidate_notes=notes_str
            )

        progress_str = self._format_progress(session.interview.progress)
        return prompts.final_system_prompt.format(
            language=session.init_info.language,
            candidate_notes=notes_str,
            challenges_history=self._format_challenges_history(session.live_coding),
            progress=progress_str
        )
    
    def build_messages(
        self,
        session: InterviewSession,
        initial_instruction: str = None
    ) -> list[dict]:
        """
        Build the messages list for OpenAI API.
        
        System prompt and internal state (notes, progress) are added fresh each time.
        Only conversation messages (user/assistant) are from session.messages.
        
        Args:
            session: The interview session
            initial_instruction: Optional instruction to add as first user message
                                 (not stored in session, used for interview start)
        """
        messages = []
        
        # System prompt (includes current state of notes and progress)
        messages.append({
            "role": "system",
            "content": self.build_system_prompt(session)
        })
        
        # Add context summary if exists
        if session.context_summary:
            messages.append({
                "role": "system",
                "content": f"[CONVERSATION SUMMARY]\n{session.context_summary}"
            })
        
        # Add initial instruction if provided (for interview start)
        if initial_instruction and not session.messages:
            messages.append({
                "role": "user",
                "content": initial_instruction
            })
        
        # Add conversation messages (user and assistant only)
        for msg in session.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return messages
    
    async def create_response(
        self,
        session: InterviewSession,
        include_tools: bool = True,
        initial_instruction: str = None,
        extra_messages: list[dict] | None = None
    ) -> dict:
        """
        Create a single response from the LLM (non-streaming).

        Returns:
        {
            "content": str,
            "tool_calls": list[{id, name, arguments}],
            "usage": {prompt_tokens, completion_tokens, total_tokens}
        }
        """
        messages = self.build_messages(session, initial_instruction)
        if extra_messages:
            messages.extend(extra_messages)
        tools = self.get_tools_definition(session) if include_tools else None

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=False,
            extra_body={
                "reasoning": {
                    "effort": "high"
                }
            }
        )

        message = response.choices[0].message
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name if tc.function else "",
                    "arguments": tc.function.arguments if tc.function else ""
                })

        return {
            "content": message.content or "",
            "tool_calls": tool_calls,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    
    
    async def summarize_context(self, messages_to_summarize: list[dict]) -> dict:
        """Summarize a portion of the conversation (chat only, not notes/plan)."""
        prompts = get_prompts()
        
        conversation_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages_to_summarize
            if msg.get('content')
        ])
        
        summary_prompt = prompts.summarization_prompt.format(
            conversation=conversation_text
        )
        
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        return {
            "summary": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
