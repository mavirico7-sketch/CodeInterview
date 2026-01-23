"""
Client for the external code executor service.
"""

from typing import Optional

import httpx

from app.config import get_settings
from app.models.session import Environment, ExecutionResult


class CodeExecutorClient:
    """HTTP client for code executor service."""

    def __init__(self):
        settings = get_settings()
        self._base_url = settings.code_executor.base_url.rstrip("/")
        self._timeout = settings.code_executor.timeout_seconds

    async def list_environments(self) -> list[Environment]:
        """Fetch available execution environments."""
        url = f"{self._base_url}/api/v1/environments"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

        environments = []
        for item in data:
            print(f"Item: {item}")
            name = item.get("name") or item.get("id") or ""
            environments.append(
                Environment(
                    id=item.get("id") or name,
                    name=name,
                    description=item.get("description", ""),
                    file_extension=item.get("file_extension", "")
                )
            )
        return environments

    async def execute_code(
        self,
        environment_name: str,
        code: str,
        stdin: Optional[str] = None,
        filename: Optional[str] = None
    ) -> ExecutionResult:
        """Execute code and return the result."""
        url = f"{self._base_url}/api/v1/execute"
        payload = {
            "environment": environment_name,
            "code": code,
        }
        if stdin is not None:
            payload["stdin"] = stdin
        if filename is not None:
            payload["filename"] = filename

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        return ExecutionResult(
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            exit_code=data.get("exit_code", 0),
            execution_time=data.get("execution_time", 0.0),
            status=data.get("status", "")
        )

