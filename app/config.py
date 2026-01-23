"""
Configuration management for the Interview Simulator.
Reads all settings from environment variables (set via interview.conf).
Prompts are loaded separately from YAML.
"""

import os
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass

import yaml
from pydantic import BaseModel


def get_env(key: str, default: str = "") -> str:
    """Get environment variable value."""
    return os.getenv(key, default)


def get_env_int(key: str, default: int) -> int:
    """Get environment variable as integer."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_env_float(key: str, default: float) -> float:
    """Get environment variable as float."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class LLMConfig:
    """LLM configuration from environment."""
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_response_tokens: int

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_key=get_env("OPENAI_API_KEY"),
            base_url=get_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=get_env("OPENAI_MODEL", "gpt-4o"),
            temperature=get_env_float("OPENAI_TEMPERATURE", 0.2),
            max_response_tokens=get_env_int("OPENAI_MAX_TOKENS", 4096),
        )


@dataclass
class MongoDBConfig:
    """MongoDB configuration from environment."""
    host: str
    port: int
    database: str
    collection: str
    username: str
    password: str

    @classmethod
    def from_env(cls) -> "MongoDBConfig":
        return cls(
            host=get_env("MONGODB_HOST", "mongodb"),
            port=get_env_int("MONGODB_PORT", 27017),
            database=get_env("MONGODB_DATABASE", "interview_simulator"),
            collection=get_env("MONGODB_COLLECTION", "sessions"),
            username=get_env("MONGODB_USERNAME", ""),
            password=get_env("MONGODB_PASSWORD", ""),
        )

    @property
    def connection_string(self) -> str:
        if self.username and self.password:
            return f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"mongodb://{self.host}:{self.port}"


@dataclass
class InterviewConfig:
    """Interview settings from environment."""
    context_token_limit: int
    total_token_limit: int
    max_exchanges: int  # Maximum number of message exchanges (user + assistant pairs)
    preserve_initial_messages: int

    @classmethod
    def from_env(cls) -> "InterviewConfig":
        return cls(
            context_token_limit=get_env_int("INTERVIEW_CONTEXT_TOKEN_LIMIT", 50000),
            total_token_limit=get_env_int("INTERVIEW_TOTAL_TOKEN_LIMIT", 1000000),
            max_exchanges=get_env_int("INTERVIEW_MAX_EXCHANGES", 25),
            preserve_initial_messages=get_env_int("INTERVIEW_PRESERVE_INITIAL_MESSAGES", 2),
        )


@dataclass
class LiveCodingConfig:
    """Live coding settings from environment."""
    max_challenges: int

    @classmethod
    def from_env(cls) -> "LiveCodingConfig":
        return cls(
            max_challenges=get_env_int("MAX_CHALLENGES", 3),
        )


@dataclass
class CodeExecutorConfig:
    """Code executor service configuration from environment."""
    base_url: str
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "CodeExecutorConfig":
        return cls(
            base_url=get_env("CODE_EXECUTOR_BASE_URL", "http://code-executor-api:8000/"),
            timeout_seconds=get_env_float("CODE_EXECUTOR_TIMEOUT_SECONDS", 15.0),
        )


@dataclass
class ServerConfig:
    """Server configuration from environment."""
    host: str
    port: int

    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls(
            host=get_env("SERVER_HOST", "0.0.0.0"),
            port=get_env_int("SERVER_PORT", 8000),
        )


@dataclass
class Settings:
    """All application settings."""
    llm: LLMConfig
    mongodb: MongoDBConfig
    interview: InterviewConfig
    live_coding: LiveCodingConfig
    code_executor: CodeExecutorConfig
    server: ServerConfig

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm=LLMConfig.from_env(),
            mongodb=MongoDBConfig.from_env(),
            interview=InterviewConfig.from_env(),
            live_coding=LiveCodingConfig.from_env(),
            code_executor=CodeExecutorConfig.from_env(),
            server=ServerConfig.from_env(),
        )


# ============================================
# Prompts Configuration (from YAML)
# ============================================

class ToolConfig(BaseModel):
    description: str


class ToolsConfig(BaseModel):
    add_candidate_note: ToolConfig
    delete_candidate_note: ToolConfig
    edit_candidate_note: ToolConfig
    change_phase: ToolConfig
    change_challenge: ToolConfig
    edit_code: ToolConfig
    execute_code: ToolConfig


class PromptsConfig(BaseModel):
    interview_system_prompt: str
    live_coding_system_prompt: str
    final_system_prompt: str
    summarization_prompt: str
    phase_transition_prompt: str
    tools: ToolsConfig


def load_prompts_yaml() -> dict:
    """Load prompts from YAML file."""
    prompts_path = Path(__file__).parent / "prompts.yaml"
    
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    
    with open(prompts_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ============================================
# Cached accessors
# ============================================

@lru_cache()
def get_settings() -> Settings:
    """Load and cache settings from environment."""
    return Settings.from_env()


@lru_cache()
def get_prompts() -> PromptsConfig:
    """Load and cache prompts from YAML."""
    config_data = load_prompts_yaml()
    return PromptsConfig(**config_data)
