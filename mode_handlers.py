"""
Execution mode handlers.
This module implements handlers for different execution modes:
- Task Mode: Generate plan and wait for user approval before execution
- Ask Mode: Answer questions without executing any commands
- Auto Mode: Generate and execute plan automatically without approval
"""

import os
import asyncio
from typing import List, Tuple, Dict, Any, Optional

from config import COMMAND_TIMEOUT, LANGUAGE_NAMES
from ai_brain import generate_execution_plan, generate_answer
from utils.terminal_utils import check_cancellation, wait_for_command_completion
from utils.logger import get_logger
from utils.messaging import send_agent_message

async def execute_commands(
    sid: str,
    plan: List[Tuple[str, str]],
    sio,
    master_fd: int,
    output_captures: Dict,
    cancellation_requests: Dict
) -> List[Dict[str, Any]]:
    """Execute a list of commands sequentially.
    
    Args:
        sid: Session ID.
        plan: List of (command, explanation) tuples to execute.
        sio: Socket.IO server instance.
        master_fd: Master file descriptor for PTY.
        output_captures: Dictionary of output captures by session ID.
        cancellation_requests: Dictionary of cancellation flags by session ID.
    
    Returns:
        List of execution records with command details.
    """
    logger = get_logger()
    executed_commands = []
    
    for command, explanation in plan:
        # Check for cancellation before each command
        if await check_cancellation(sid, sio, cancellation_requests, {}, master_fd):
            break
        
        # Start capturing output
        if not output_captures[sid].active:
            output_captures[sid].start()
        
        await asyncio.sleep(0.3)
        
        # Clear buffer for this command
        output_captures[sid].buffer = ""
        output_captures[sid].last_output_time = asyncio.get_event_loop().time()
        
        # Execute command
        os.write(master_fd, (command + '\n').encode())
        
        # Wait for completion with cancellation checks
        await wait_for_command_completion(
            sid,
            output_captures,
            timeout=COMMAND_TIMEOUT,
            cancellation_requests=cancellation_requests
        )
        
        # Record execution
        executed_commands.append(
            logger.log_command_execution(sid, command, explanation)
        )
        
        # Check if cancelled during execution
        if cancellation_requests.get(sid, False):
            break
    
    # Stop capturing
    if sid in output_captures:
        output_captures[sid].stop_and_get()
    
    return executed_commands


async def handle_ask_mode(
    sid: str,
    query: str,
    context: str,
    sio,
    master_fd: int,
    output_captures: Dict,
    cancellation_requests: Dict,
    model: Optional[str] = None,
    language: str = "en"
) -> None:
    """Handle Ask mode - provide answer without command execution.
    
    Args:
        sid: Session ID.
        query: User's question.
        context: Terminal context.
        sio: Socket.IO server instance.
        master_fd: Master file descriptor for PTY.
        output_captures: Dictionary of output captures.
        cancellation_requests: Dictionary of cancellation flags.
        model: Model to use ('gemini' or 'qwen').
        language: Language code for response.
    """
    logger = get_logger()
    language_name = LANGUAGE_NAMES.get(language, 'English')
    
    answer = await generate_answer(query, context, model=model, language=language)
    
    # Check for cancellation
    if await check_cancellation(sid, sio, cancellation_requests, {}, master_fd):
        logger.log_interaction(sid, query, "ask", answer=answer, status="cancelled", 
                             model=model, language=language_name)
        cancellation_requests[sid] = False
        return
    
    # Send answer to chat panel
    await send_agent_message(sio, sid, answer if answer else "❌ Could not generate an answer.")
    
    logger.log_interaction(sid, query, "ask", answer=answer, status="completed",
                         model=model, language=language_name)

async def handle_auto_mode(
    sid: str,
    query: str,
    context: str,
    sio,
    master_fd: int,
    output_captures: Dict,
    cancellation_requests: Dict,
    model: Optional[str] = None,
    language: str = "en"
) -> None:
    """Handle Auto mode - create and execute plan automatically.
    
    Args:
        sid: Session ID
        query: User's task or goal
        context: Terminal context
        sio: Socket.IO server instance
        master_fd: Master file descriptor for PTY
        output_captures: Dictionary of output captures
        cancellation_requests: Dictionary of cancellation flags
        model: Model to use ('gemini' or 'qwen')
        language: Language code for response
    """
    logger = get_logger()
    language_name = LANGUAGE_NAMES.get(language, 'English')
    
    plan = await generate_execution_plan(query, context, model=model, language=language)
    
    # Check for cancellation after plan generation
    if await check_cancellation(sid, sio, cancellation_requests, {}, master_fd):
        logger.log_interaction(
            sid, query, "auto",
            plan=[{"command": c, "explanation": e} for c, e in (plan or [])],
            executed_commands=[],
            status="cancelled",
            model=model,
            language=language_name
        )
        cancellation_requests[sid] = False
        return
    
    if not plan:
        await send_agent_message(sio, sid, "❌ Agent could not form a plan.")
        logger.log_interaction(
            sid, query, "auto",
            status="error",
            error="Could not form a plan",
            model=model,
            language=language_name
        )
        return
    
    # Format and send plan to chat
    plan_md = f"## 🚀 Auto Mode - Execution Plan\n\n**Goal:** {query}\n\n### Execution Steps\n\n"
    for i, (command, explanation) in enumerate(plan, 1):
        plan_md += f"**{i}.** `{command}`\n\n_{explanation}_\n\n"
    plan_md += "\n⚡ **Executing automatically...**"
    
    await send_agent_message(sio, sid, plan_md)
    
    # Check for cancellation before execution
    if await check_cancellation(sid, sio, cancellation_requests, {}, master_fd):
        logger.log_interaction(
            sid, query, "auto",
            plan=[{"command": c, "explanation": e} for c, e in plan],
            executed_commands=[],
            status="cancelled",
            model=model,
            language=language_name
        )
        cancellation_requests[sid] = False
        return
    
    await asyncio.sleep(0.5)
    
    # Execute all commands
    executed_commands = await execute_commands(
        sid, plan, sio, master_fd, output_captures, cancellation_requests
    )
    
    # Send completion message to chat
    status = "cancelled" if cancellation_requests.get(sid) else "completed"
    if status == "completed":
        await send_agent_message(
            sio, sid,
            f"✅ **Execution completed!** ({len(executed_commands)}/{len(plan)} commands executed)"
        )
    else:
        await send_agent_message(
            sio, sid,
            f"🛑 **Execution cancelled.** ({len(executed_commands)}/{len(plan)} commands executed)"
        )
    
    logger.log_interaction(
        sid, query, "auto",
        plan=[{"command": c, "explanation": e} for c, e in plan],
        executed_commands=executed_commands,
        status=status,
        model=model,
        language=language_name
    )
    
    # Reset cancellation flag after execution completes
    cancellation_requests[sid] = False

async def handle_task_mode(
    sid: str,
    query: str,
    context: str,
    sio,
    master_fd: int,
    output_captures: Dict,
    pending_plans: Dict,
    cancellation_requests: Dict,
    model: Optional[str] = None,
    language: str = "en"
) -> None:
    """Handle Task mode - create plan and wait for user approval.
    
    Args:
        sid: Session ID
        query: User's task or goal
        context: Terminal context
        sio: Socket.IO server instance
        master_fd: Master file descriptor for PTY
        output_captures: Dictionary of output captures
        pending_plans: Dictionary of pending plans awaiting approval
        cancellation_requests: Dictionary of cancellation flags
        model: Model to use ('gemini' or 'qwen')
        language: Language code for response
    """
    logger = get_logger()
    language_name = LANGUAGE_NAMES.get(language, 'English')
    
    plan = await generate_execution_plan(query, context, model=model, language=language)
    
    # Check for cancellation after plan generation
    if await check_cancellation(sid, sio, cancellation_requests, {}, master_fd):
        logger.log_interaction(
            sid, query, "task",
            plan=[{"command": c, "explanation": e} for c, e in (plan or [])],
            status="cancelled",
            model=model,
            language=language_name
        )
        cancellation_requests[sid] = False
        return
    
    if not plan:
        await send_agent_message(sio, sid, "❌ Agent could not form a plan.")
        logger.log_interaction(
            sid, query, "task",
            status="error",
            error="Could not form a plan",
            model=model,
            language=language_name
        )
        return
    
    # Format and send plan to chat
    plan_md = f"## 📝 Task Mode - Execution Plan\n\n**Goal:** {query}\n\n### Proposed Steps\n\n"
    for i, (command, explanation) in enumerate(plan, 1):
        plan_md += f"**{i}.** `{command}`\n\n_{explanation}_\n\n"
    plan_md += "\n⏳ **Waiting for your approval...**"
    
    await send_agent_message(sio, sid, plan_md)
    
    # Check for cancellation before storing pending plan
    if await check_cancellation(sid, sio, cancellation_requests, {}, master_fd):
        logger.log_interaction(
            sid, query, "task",
            plan=[{"command": c, "explanation": e} for c, e in plan],
            status="cancelled",
            model=model,
            language=language_name
        )
        cancellation_requests[sid] = False
        return
    
    # Store plan for approval
    pending_plans[sid] = {
        'plan': plan,
        'query': query,
        'context': context,
        'model': model,
        'language': language_name
    }
    
    # Show approval buttons and stop loading indicator
    await sio.emit('agent_working', {'working': False}, room=sid)
    await sio.emit('show_approval_buttons', {'show': True})

