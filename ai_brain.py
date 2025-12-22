"""
AI Brain - Language model integration for intelligent terminal operations.
This module provides the core AI functionality for generating execution plans,
making iterative decisions, and answering user questions in natural language.
"""

import json
from typing import Optional, List, Tuple, Dict, Any

from config import PROMPTS, LANGUAGE_NAMES

# Global model instance (set during application startup)
_model = None


def configure_model(model) -> None:
    """Configure the global AI model instance.
    
    Args:
        model: The AI model instance to use for all generation tasks.
    """
    global _model
    _model = model


async def generate_execution_plan(
    query: str,
    context: str,
    execution_history: str = "None",
    model: Optional[str] = None,
    language: str = "en"
) -> Optional[List[Tuple[str, str]]]:
    """Generate an execution plan from user query and terminal context.
    
    Args:
        query: User's task or goal description.
        context: Current terminal screen context.
        execution_history: History of previous command executions.
        model: Optional model identifier ('gemini' or 'qwen').
        language: Language code for the response (default: 'en').
    
    Returns:
        List of (command, explanation) tuples, or None if generation fails.
    """
    language_name = LANGUAGE_NAMES.get(language, 'English')
    prompt = PROMPTS['task'].format(
        user_query=query,
        screen_context=context,
        execution_history=execution_history,
        language=language_name
    )
    
    try:
        response = await _model.generate_content_async(prompt, model=model)
        
        # Clean and parse JSON response
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned_text)
        
        plan_steps = data.get("plan", [])
        if plan_steps:
            return [(step["command"], step["explanation"]) for step in plan_steps]
        
        return None
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse AI response as JSON: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to generate execution plan: {e}")
        return None

async def generate_first_iterative_command(
    query: str,
    context: str,
    model: Optional[str] = None,
    language: str = "en"
) -> Dict[str, Any]:
    """Generate the first command to execute in iterative mode.
    
    Args:
        query: User's task or goal description.
        context: Current terminal screen context.
        model: Optional model identifier ('gemini' or 'qwen').
        language: Language code for the response (default: 'en').
    
    Returns:
        Dictionary with 'command' and 'explanation' keys.
    """
    language_name = LANGUAGE_NAMES.get(language, 'English')
    prompt = PROMPTS['iterative_first'].format(
        user_query=query,
        screen_context=context,
        language=language_name
    )
    
    try:
        response = await _model.generate_content_async(prompt, model=model)
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        # Extract JSON from response text
        json_start = cleaned_text.find('{')
        json_end = cleaned_text.rfind('}')
        
        if json_start != -1 and json_end != -1:
            cleaned_text = cleaned_text[json_start:json_end + 1]
        
        return json.loads(cleaned_text)
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse first iterative command: {e}")
        return {
            "command": None,
            "explanation": f"Failed to parse AI response: {str(e)}"
        }
    except Exception as e:
        print(f"[ERROR] Failed to generate first iterative command: {e}")
        return {
            "command": None,
            "explanation": f"Failed to get AI response: {str(e)}"
        }


async def generate_iterative_decision(
    query: str,
    previous_commands: str,
    latest_output: str,
    model: Optional[str] = None,
    language: str = "en"
) -> Dict[str, Any]:
    """Determine the next action in iterative mode based on command output.
    
    Args:
        query: Original user's task or goal.
        previous_commands: History of executed commands.
        latest_output: Output from the last executed command.
        model: Optional model identifier ('gemini' or 'qwen').
        language: Language code for the response (default: 'en').
    
    Returns:
        Dictionary with 'continue', 'next_command', 'explanation', and 'status' keys.
    """
    language_name = LANGUAGE_NAMES.get(language, 'English')
    prompt = PROMPTS['iterative'].format(
        user_query=query,
        previous_commands=previous_commands,
        latest_output=latest_output,
        language=language_name
    )
    
    try:
        response = await _model.generate_content_async(prompt, model=model)
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        # Extract JSON from response text
        json_start = cleaned_text.find('{')
        json_end = cleaned_text.rfind('}')
        
        if json_start != -1 and json_end != -1:
            cleaned_text = cleaned_text[json_start:json_end + 1]
        
        return json.loads(cleaned_text)
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse iterative decision: {e}")
        return {
            "continue": False,
            "status": "error",
            "explanation": f"Failed to parse AI response: {str(e)}"
        }
    except Exception as e:
        print(f"[ERROR] Failed to generate iterative decision: {e}")
        return {
            "continue": False,
            "status": "error",
            "explanation": f"Failed to get AI response: {str(e)}"
        }


async def generate_answer(
    query: str,
    context: str,
    model: Optional[str] = None,
    language: str = "en"
) -> str:
    """Generate a natural language answer to a user question.
    
    Args:
        query: User's question.
        context: Current terminal screen context.
        model: Optional model identifier ('gemini' or 'qwen').
        language: Language code for the response (default: 'en').
    
    Returns:
        AI-generated answer as markdown text.
    """
    language_name = LANGUAGE_NAMES.get(language, 'English')
    prompt = PROMPTS['ask'].format(
        user_query=query,
        screen_context=context,
        language=language_name
    )
    
    try:
        response = await _model.generate_content_async(prompt, model=model)
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Failed to generate answer: {e}")
        return "I encountered an error while processing your question. Please try again."
