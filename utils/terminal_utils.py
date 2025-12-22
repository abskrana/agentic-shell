"""Terminal utilities for output capture and command monitoring.

Provides utilities for capturing terminal output and monitoring command
execution completion using various heuristics and pattern matching.
"""

import os
import asyncio
import re
from typing import Dict, Optional

from config import NO_OUTPUT_THRESHOLD, MIN_WAIT_TIME, PROMPT_CONFIRMATION_DELAY


class OutputCapture:
    """Captures and manages terminal output for a session."""
    
    def __init__(self) -> None:
        """Initialize output capture with empty buffer."""
        self.buffer = ""
        self.active = False
        self.last_output_time = asyncio.get_event_loop().time()
    
    def start(self) -> None:
        """Start capturing output."""
        self.buffer = ""
        self.active = True
        self.last_output_time = asyncio.get_event_loop().time()
    
    def append(self, data: str) -> None:
        """Append data to capture buffer if active.
        
        Args:
            data: Terminal output data to append.
        """
        if self.active:
            self.buffer += data
            self.last_output_time = asyncio.get_event_loop().time()
    
    def stop_and_get(self) -> str:
        """Stop capturing and return captured output.
        
        Returns:
            The captured output buffer.
        """
        self.active = False
        result = self.buffer
        self.buffer = ""
        return result
    
    def has_recent_output(self, threshold: float = 1.0) -> bool:
        """Check if output was received recently.
        
        Args:
            threshold: Time threshold in seconds.
        
        Returns:
            True if output was received within the threshold time.
        """
        return (asyncio.get_event_loop().time() - self.last_output_time) < threshold
    
    def looks_like_prompt(self) -> bool:
        """Check if buffer ends with what looks like a shell prompt.
        
        Uses multiple heuristics to detect common shell prompt patterns,
        including standard bash prompts, cloud shell prompts, and prompts
        with project/context information.
        
        Returns:
            True if the buffer appears to end with a shell prompt.
        """
        if not self.buffer:
            return False
        
        # Get the last non-empty line
        lines = [line for line in self.buffer.split('\n') if line.strip()]
        if not lines:
            return False
        
        last_line = lines[-1]
        
        # Remove ANSI escape sequences for better detection
        clean_line = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', last_line)
        clean_line = re.sub(r'\r', '', clean_line).strip()
        
        # Prevent false positives on very long lines
        max_prompt_length = 300
        
        # Pattern 1: Standard prompt ending with $ or #
        if clean_line.endswith(('$', '$ ', '#', '# ')):
            if '/' in clean_line or ('@' in clean_line and ':' in clean_line):
                return len(clean_line) < max_prompt_length
        
        # Pattern 2: Cloud Shell format with project ID
        if re.search(r'[@:].*\$.*\([^)]+\)\s*$', clean_line):
            return len(clean_line) < max_prompt_length
        
        # Pattern 3: Prompt with project/context in parentheses
        if re.search(r'[@:].+\([^)]+\)\s*$', clean_line):
            if '@' in clean_line and (':' in clean_line or '~' in clean_line):
                return len(clean_line) < max_prompt_length
        
        # Pattern 4: Generic prompt pattern with @ and $
        if '@' in clean_line and re.search(r'[$#]', clean_line):
            return len(clean_line) < max_prompt_length
        
        return False

async def check_cancellation(
    sid: str,
    sio,
    cancellation_requests: Dict[str, bool],
    iterative_sessions: Dict,
    master_fd: int,
    cleanup_session: bool = False
) -> bool:
    """Check if user requested cancellation.
    
    Args:
        sid: Session ID.
        sio: Socket.IO server instance.
        cancellation_requests: Dictionary of cancellation flags.
        iterative_sessions: Dictionary of iterative session data.
        master_fd: Master file descriptor for PTY.
        cleanup_session: Whether to cleanup iterative session on cancellation.
    
    Returns:
        True if cancelled, False otherwise.
    """
    if cancellation_requests.get(sid, False):
        from utils.messaging import send_agent_message
        await send_agent_message(sio, sid, '🛑 **Cancelled by user.**')
        
        if cleanup_session and sid in iterative_sessions:
            iterative_sessions[sid]['active'] = False
            del iterative_sessions[sid]
        
        return True
    
    return False

async def wait_for_command_completion(
    sid: str,
    output_captures: Dict[str, OutputCapture],
    timeout: int = 60,
    cancellation_requests: Optional[Dict[str, bool]] = None,
    iterative_sessions: Optional[Dict] = None
) -> bool:
    """Wait for a command to complete by monitoring output patterns.
    
    Uses multiple heuristics to detect when a command has finished:
    - Prompt detection (looks_like_prompt)
    - Output stability (no changes in buffer)
    - Output recency (no new output for threshold time)
    
    Args:
        sid: Session ID.
        output_captures: Dictionary of output captures by session ID.
        timeout: Maximum time to wait in seconds.
        cancellation_requests: Optional dict to check for cancellation.
        iterative_sessions: Optional dict to mark session inactive on cancellation.
    
    Returns:
        True if command completed normally, False if cancelled/timeout.
    """
    from config import execution as execution_config
    
    start_time = asyncio.get_event_loop().time()
    
    # Initial minimum wait time
    await asyncio.sleep(MIN_WAIT_TIME)
    
    prompt_detected_time = None
    stable_buffer_count = 0
    last_buffer_content = ""
    
    while True:
        current_time = asyncio.get_event_loop().time()
        
        # Check for cancellation
        if cancellation_requests and cancellation_requests.get(sid, False):
            if iterative_sessions and sid in iterative_sessions:
                iterative_sessions[sid]['active'] = False
            return False
        
        # Check timeout
        if current_time - start_time > timeout or sid not in output_captures:
            return sid not in output_captures
        
        capture = output_captures[sid]
        current_buffer = capture.buffer
        
        # Check buffer stability
        if current_buffer == last_buffer_content and current_buffer:
            stable_buffer_count += 1
        else:
            stable_buffer_count = 0
            last_buffer_content = current_buffer
        
        # Buffer stable and looks like prompt
        if stable_buffer_count >= 2 and capture.looks_like_prompt():
            if not capture.has_recent_output(0.5):
                return True
        
        # Prompt detection with confirmation delay
        if capture.looks_like_prompt():
            if prompt_detected_time is None:
                prompt_detected_time = current_time
            elif (current_time - prompt_detected_time) >= PROMPT_CONFIRMATION_DELAY:
                if not capture.has_recent_output(0.5):
                    return True
        else:
            prompt_detected_time = None
        
        # Fallback: no recent output and looks like prompt
        if not capture.has_recent_output(NO_OUTPUT_THRESHOLD) and capture.looks_like_prompt():
            await asyncio.sleep(0.5)
            if not capture.has_recent_output(0.3):
                return True
        
        # Determine check interval
        check_interval = execution_config.cancellation_check_interval if cancellation_requests else 0.3
        await asyncio.sleep(check_interval)
