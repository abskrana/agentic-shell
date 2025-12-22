"""Logging module with cloud storage integration.

Provides structured logging capabilities with automatic cloud storage
through the Lightning AI backend.
"""

import json
import aiohttp
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional


class AgentLogger:
    """Structured logger with Lightning AI cloud storage integration."""
    
    def __init__(self, lightning_api_url: str):
        """Initialize logger with cloud storage endpoint.
        
        Args:
            lightning_api_url: Lightning AI API endpoint for log storage.
        """
        self.lightning_api_url = lightning_api_url
    
    def log_interaction(
        self,
        session_id: str,
        prompt: str,
        mode: str,
        plan: Optional[List[Dict]] = None,
        executed_commands: Optional[List[Dict]] = None,
        answer: Optional[str] = None,
        status: str = "completed",
        error: Optional[str] = None,
        metadata: Optional[Dict] = None,
        model: Optional[str] = None,
        language: Optional[str] = None
    ) -> None:
        """Log a complete agent interaction session.
        
        Args:
            session_id: Unique session identifier.
            prompt: User's input prompt.
            mode: Execution mode (task/ask/auto/iterative).
            plan: Optional execution plan.
            executed_commands: Optional list of executed commands.
            answer: Optional answer for ask mode.
            status: Status of the interaction (completed/error/cancelled).
            error: Optional error message.
            metadata: Optional additional metadata (not logged).
            model: Model used (gemini/qwen).
            language: Language used for response.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": session_id,
            "prompt": prompt,
            "mode": mode,
            "status": status,
        }
        
        # Add optional fields only if provided
        if plan is not None:
            entry["plan"] = plan
        if executed_commands is not None:
            entry["executed_commands"] = executed_commands
        if answer is not None:
            entry["answer"] = answer
        if error is not None:
            entry["error"] = error
        if model is not None:
            entry["model"] = model
        if language is not None:
            entry["language"] = language
        
        self._write_log(entry)
    
    def log_command_execution(
        self,
        session_id: str,
        command: str,
        explanation: str
    ) -> Dict[str, Any]:
        """Create a command execution record.
        
        Args:
            session_id: Unique session identifier.
            command: The executed command.
            explanation: Explanation of the command's purpose.
        
        Returns:
            Dictionary containing command execution details.
        """
        return {
            "command": command,
            "explanation": explanation,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    def _write_log(self, entry: Dict[str, Any]) -> None:
        """Write a log entry to cloud storage.
        
        Args:
            entry: Log entry dictionary to write.
        """
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._send_to_cloud(entry))
        except RuntimeError:
            print("[WARNING] No event loop available for cloud logging")
    
    async def _send_to_cloud(
        self,
        entry: Dict[str, Any],
        max_retries: int = 3
    ) -> None:
        """Send log entry to Lightning AI storage with retry logic.
        
        Args:
            entry: Log entry to send.
            max_retries: Maximum number of retry attempts.
        """
        headers = {'Content-Type': 'application/json'}
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.lightning_api_url,
                        json=entry,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status in (200, 201, 204):
                            return
                        elif response.status >= 500 and attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        else:
                            raise Exception(f"API returned status {response.status}")
                            
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[WARNING] Failed to send log after {max_retries} attempts: {e}")
                    return
                await asyncio.sleep(2 ** attempt)


# Global logger singleton
_logger = None


def initialize_logger(
    lightning_api_url: str,
    force_new: bool = False
) -> AgentLogger:
    """Initialize or retrieve the global logger instance.
    
    Args:
        lightning_api_url: Lightning AI API endpoint for log storage.
        force_new: If True, create a new logger instance.
    
    Returns:
        The global AgentLogger instance.
    """
    global _logger
    
    if _logger is None or force_new:
        _logger = AgentLogger(lightning_api_url=lightning_api_url)
    
    return _logger


def get_logger() -> AgentLogger:
    """Get the global logger instance.
    
    Returns:
        The global AgentLogger instance.
    
    Raises:
        RuntimeError: If logger has not been initialized.
    """
    global _logger
    
    if _logger is None:
        raise RuntimeError(
            "Logger not initialized. Call initialize_logger() first."
        )
    
    return _logger
