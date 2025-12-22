"""
Application configuration module.
Centralized configuration management for the Agentic Shell application using dataclasses.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ServerConfig:
    """Server configuration settings."""
    host: str = field(default_factory=lambda: os.getenv('HOST', '0.0.0.0'))
    port: int = field(default_factory=lambda: int(os.getenv('PORT', 8088)))


@dataclass(frozen=True)
class ExecutionConfig:
    """Command execution configuration."""
    command_timeout: int = 60
    iterative_timeout: int = 300
    max_iterations: int = 50
    no_output_threshold: float = 3.0
    min_wait_time: float = 0.5
    prompt_confirmation_delay: float = 0.5
    cancellation_check_interval: float = 0.3


@dataclass(frozen=True)
class SpeechConfig:
    """Speech recognition configuration."""
    
    languages: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        'off': {'code': None, 'name': 'Off'},
        'en': {'code': 'en-IN', 'name': 'English'},
        'hi': {'code': 'hi-IN', 'name': 'Hindi'},
        'bn': {'code': 'bn-IN', 'name': 'Bengali'},
        'te': {'code': 'te-IN', 'name': 'Telugu'},
        'mr': {'code': 'mr-IN', 'name': 'Marathi'},
        'ta': {'code': 'ta-IN', 'name': 'Tamil'},
        'gu': {'code': 'gu-IN', 'name': 'Gujarati'},
        'kn': {'code': 'kn-IN', 'name': 'Kannada'},
        'ml': {'code': 'ml-IN', 'name': 'Malayalam'},
        'pa': {'code': 'pa-IN', 'name': 'Punjabi'},
        'or': {'code': 'or-IN', 'name': 'Odia'},
        'as': {'code': 'as-IN', 'name': 'Assamese'},
    })
    
    context_phrases: List[str] = field(default_factory=lambda: [
        "create", "delete", "remove", "list", "show", "install", "run", "execute",
        "directory", "folder", "file", "script", "command", "terminal",
        "Python", "JavaScript", "HTML", "CSS", "Git", "Docker", "Kubernetes",
        "mkdir", "rmdir", "ls", "cd", "pwd", "cat", "grep", "find", "chmod",
        "pip install", "npm install", "git clone", "git commit", "git push",
    ])
    
    context_boost: float = 15.0


@dataclass(frozen=True)
class LightningAIConfig:
    """Lightning AI backend configuration."""
    unified_backend_url: str = field(default_factory=lambda: os.getenv('LIGHTNING_UNIFIED_URL', ''))


# Create immutable configuration instances
server = ServerConfig()
execution = ExecutionConfig()
speech = SpeechConfig()
lightning = LightningAIConfig()

# Convenience exports for backward compatibility
HOST = server.host
PORT = server.port
COMMAND_TIMEOUT = execution.command_timeout
NO_OUTPUT_THRESHOLD = execution.no_output_threshold
MIN_WAIT_TIME = execution.min_wait_time
PROMPT_CONFIRMATION_DELAY = execution.prompt_confirmation_delay
SUPPORTED_LANGUAGES = speech.languages
SPEECH_CONTEXT_PHRASES = speech.context_phrases
SPEECH_CONTEXT_BOOST = speech.context_boost
MAX_ITERATIONS = execution.max_iterations

# AI Prompt Templates
PROMPTS = {
    'task': """You are an expert Bash command line assistant integrated into a user's terminal.
Your goal is to create a sequence of commands to achieve the user's goal.
You are given the user's high-level goal and the current context of their terminal screen. Use the screen context to inform your plan.

**SCREEN CONTEXT:**

{screen_context}

**USER'S GOAL:**
"{user_query}"

**EXECUTION HISTORY:**
{execution_history}

**CONSTRAINTS:**
- Output ONLY a valid JSON object with a single key: `"plan"`.
- The `"plan"` value must be a LIST of objects.
- Each object must have `command` and `explanation` keys.
- If the context shows an error, your plan should be to fix it.
- `cd` must be its own step.
- For multi-line file content, use heredoc syntax: cat > file.txt << 'EOF'\\ncontent\\nEOF
- For creating files with content, prefer heredoc over echo when content has multiple lines
- Always use absolute paths or navigate to correct directory first
- If previous execution history shows errors, adjust your plan accordingly
- Respond in {language} language. Write explanations for the user in {language}, but keep commands in English/Bash syntax.

**Your JSON Response:**
""",
    
    'iterative_first': """You are an expert Bash command line assistant starting an iterative execution session.
The user has a goal, and you need to determine the FIRST command to execute.

**SCREEN CONTEXT:**

{screen_context}

**USER'S GOAL:**
"{user_query}"

**TASK:**
Analyze the goal and context, then decide what the FIRST command should be to start working towards this goal.

**CONSTRAINTS:**
- Output ONLY a valid JSON object with keys: `"command"`, `"explanation"`
- `"command"`: string (the first bash command to execute)
- `"explanation"`: string (explain to the user why this is the right first step)
- For multi-line file content, use heredoc syntax: cat > file.txt << 'EOF'\\ncontent\\nEOF
- Always use absolute paths or navigate to correct directory first
- Respond in {language} language. Write explanations in {language}, but keep commands in English/Bash syntax.

**Your JSON Response:**
""",
    
    'iterative': """You are an expert Bash command line assistant in an iterative execution mode.
You have executed some commands and received output. Based on the output, decide what to do next.

**ORIGINAL GOAL:**
"{user_query}"

**PREVIOUS COMMANDS EXECUTED:**
{previous_commands}

**LATEST OUTPUT:**
{latest_output}

**TASK:**
Analyze the output and decide the next action. If the task is complete or an error occurred that you cannot fix, set `"continue"` to false.

**CONSTRAINTS:**
- Output ONLY a valid JSON object with keys: `"continue"`, `"next_command"`, `"explanation"`, `"status"`
- `"continue"`: boolean (true if more commands needed, false if done or error)
- `"next_command"`: string (the next bash command, or empty if done)
- `"explanation"`: string (explain to the user what you observed and why you're doing this)
- `"status"`: "success", "error", or "in_progress"
- For multi-line file content, use heredoc syntax: cat > file.txt << 'EOF'\\ncontent\\nEOF
- Adjust commands based on the output you see
- Respond in {language} language. Write explanations in {language}, but keep commands in English/Bash syntax.
- When the next command is about executing something after creating it, like a program, or hosting on a url, then stop with success status, and state in the explanation that using which command user can do the execution step.
**Your JSON Response:**
""",
    
    'ask': """You are a helpful assistant with expertise in command line tools, programming, and system administration.
The user may be asking you a question in the context of their terminal session.

**SCREEN CONTEXT:**

{screen_context}

**USER'S QUESTION:**
"{user_query}"

**OUTPUT FORMAT:**
Your response will be rendered in the chat box, so use simple Markdown formatting.

- Use headings (e.g., `### Title`) for structure.
- Use bullet points (`-`) for lists.
- Use code blocks (example ```bash) for commands.
- Use **bold** for emphasis.
- Keep it clean and readable.

**CRITICAL:**
- Respond in {language} language. Write your entire response for the user in {language}, including explanations, but keep code examples and commands in their appropriate syntax (English/Bash/code).
- Do not use complex or nested Markdown. Simple is better.
- Ensure the response is a single, complete Markdown block.
"""
}

# Language name mapping for prompts
LANGUAGE_NAMES = {
    'off': 'English',
    'en': 'English',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'te': 'Telugu',
    'mr': 'Marathi',
    'ta': 'Tamil',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi',
    'or': 'Odia',
    'as': 'Assamese'
}
