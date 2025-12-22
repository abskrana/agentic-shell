"""Socket.IO event handlers for client-server communication."""

import os
import asyncio
import base64
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from config import SUPPORTED_LANGUAGES, SPEECH_CONTEXT_PHRASES, SPEECH_CONTEXT_BOOST, COMMAND_TIMEOUT, LANGUAGE_NAMES, MAX_ITERATIONS
from ai_brain import generate_iterative_decision, generate_execution_plan, generate_first_iterative_command
from utils.terminal_utils import OutputCapture, wait_for_command_completion
from utils.logger import get_logger
from mode_handlers import handle_ask_mode, handle_auto_mode, handle_task_mode, execute_commands
from utils.messaging import send_agent_message

lightning_client = None

# Session management abstractions

@dataclass
class SessionState:
    """Container for unified session state management."""
    output_capture: OutputCapture = field(default_factory=OutputCapture)
    iterative_data: Optional[dict] = None
    pending_plan: Optional[dict] = None
    cancelled: bool = False
    
    def mark_cancelled(self):
        """Mark session as cancelled and cleanup iterative state."""
        self.cancelled = True
        if self.iterative_data:
            self.iterative_data['active'] = False


class SessionManager:
    """Centralized manager for all session states."""
    
    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}
    
    def get(self, sid: str) -> SessionState:
        """Get or create session state for a given session ID."""
        if sid not in self._sessions:
            self._sessions[sid] = SessionState()
        return self._sessions[sid]
    
    def cleanup(self, sid: str):
        """Remove session state for a given session ID."""
        self._sessions.pop(sid, None)
    
    def is_cancelled(self, sid: str) -> bool:
        """Check if a session has been cancelled."""
        return self.get(sid).cancelled
    
    def cancel(self, sid: str):
        """Mark a session as cancelled."""
        self.get(sid).mark_cancelled()
    
    def reset_cancellation(self, sid: str):
        """Reset the cancellation flag for a session."""
        self.get(sid).cancelled = False


class CommandExecutor:
    """Handles command execution with output capture."""
    
    def __init__(self, master_fd: int, session_manager: SessionManager):
        self.master_fd = master_fd
        self.session_manager = session_manager
    
    async def prepare_and_execute(self, sid: str, command: str, delay: float = 0.3) -> str:
        """Prepare output capture and execute a command.
        
        Args:
            sid: Session ID.
            command: Command to execute.
            delay: Delay before execution in seconds.
            
        Returns:
            The executed command.
        """
        capture = self.session_manager.get(sid).output_capture
        
        if not capture.active:
            capture.start()
        
        await asyncio.sleep(delay)
        
        # Clear buffer for fresh output
        capture.buffer = ""
        capture.last_output_time = asyncio.get_event_loop().time()
        
        # Execute command
        os.write(self.master_fd, (command + '\n').encode())
        
        return command
    
    async def wait_for_completion(self, sid: str, timeout: int = COMMAND_TIMEOUT) -> str:
        """Wait for command completion and return output.
        
        Args:
            sid: Session ID.
            timeout: Maximum time to wait in seconds.
            
        Returns:
            The captured output.
        """
        capture = self.session_manager.get(sid).output_capture
        
        await wait_for_command_completion(
            sid, 
            {sid: capture}, 
            timeout=timeout,
            cancellation_requests={sid: self.session_manager.is_cancelled(sid)}
        )
        
        return capture.buffer
    
    def stop_capture(self, sid: str) -> str:
        """Stop output capture and return final output.
        
        Args:
            sid: Session ID.
            
        Returns:
            The final captured output.
        """
        return self.session_manager.get(sid).output_capture.stop_and_get()


def format_plan_message(mode: str, query: str, plan: List[Tuple[str, str]], footer: str = "") -> str:
    """Format execution plan as markdown for display.
    
    Args:
        mode: Execution mode ('task', 'auto', or 'iterative').
        query: User's original query.
        plan: List of (command, explanation) tuples.
        footer: Optional footer text.
        
    Returns:
        Formatted markdown string.
    """
    mode_icons = {
        'task': '📝 Task Mode',
        'auto': '🚀 Auto Mode',
        'iterative': '🔄 Iterative Mode'
    }
    
    header = f"## {mode_icons.get(mode, '⚡ Mode')} - Execution Plan\n\n"
    header += f"**Goal:** {query}\n\n"
    header += f"### {'Proposed Steps' if mode == 'task' else 'Execution Steps'}\n\n"
    
    body = ""
    for i, (command, explanation) in enumerate(plan, 1):
        body += f"**{i}.** `{command}`\n\n_{explanation}_\n\n"
    
    return header + body + "\n" + footer


async def check_and_handle_cancellation(
    sio, 
    sid: str, 
    session_manager: SessionManager, 
    logger, 
    query: str, 
    mode: str, 
    **log_data
) -> bool:
    """Check for cancellation and handle cleanup if cancelled.
    
    Args:
        sio: Socket.IO server instance.
        sid: Session ID.
        session_manager: Session manager instance.
        logger: Logger instance.
        query: User's original query.
        mode: Execution mode.
        **log_data: Additional data to log.
        
    Returns:
        True if cancelled, False otherwise.
    """
    if session_manager.is_cancelled(sid):
        await send_agent_message(sio, sid, '🛑 **Cancelled by user.**')
        logger.log_interaction(sid, query, mode, status="cancelled", **log_data)
        session_manager.reset_cancellation(sid)
        return True
    return False

# Event handler registration

def register_handlers(sio, master_fd, output_captures, pending_plans, iterative_sessions, cancellation_requests):
    """Register all Socket.IO event handlers.
    
    Args:
        sio: Socket.IO server instance.
        master_fd: Master file descriptor for PTY.
        output_captures: Dictionary of output captures by session ID.
        pending_plans: Dictionary of pending plans by session ID.
        iterative_sessions: Dictionary of iterative sessions by session ID.
        cancellation_requests: Dictionary of cancellation flags by session ID.
    """
    
    # Initialize managers
    session_manager = SessionManager()
    executor = CommandExecutor(master_fd, session_manager)
    logger = get_logger()
    
    @sio.event
    async def connect(sid, environ):
        print(f"Client connected: {sid}")
        # Initialize session state
        state = session_manager.get(sid)
        state.output_capture = OutputCapture()
        state.cancelled = False
        # Also maintain legacy dicts for mode_handlers compatibility
        output_captures[sid] = state.output_capture
        cancellation_requests[sid] = False
        os.write(master_fd, b'\n')

    @sio.event
    async def disconnect(sid):
        print(f"Client disconnected: {sid}")
        session_manager.cleanup(sid)
        # Clean up legacy dicts for mode_handlers compatibility
        output_captures.pop(sid, None)
        cancellation_requests.pop(sid, None)
        pending_plans.pop(sid, None)

    @sio.event
    async def pty_input(sid, data):
        os.write(master_fd, data['input'].encode())

    @sio.event
    async def get_supported_languages(sid):
        await sio.emit('supported_languages', {'languages': SUPPORTED_LANGUAGES}, room=sid)

    @sio.event
    async def transcribe_audio(sid, data):
        """Transcribe audio using Lightning AI Speech-to-Text."""
        try:
            if not lightning_client:
                await sio.emit('transcription_error', {
                    'error': 'Speech-to-Text service not available. Check Lightning AI configuration.'
                }, room=sid)
                return
            
            audio_data = data.get('audio')
            if not audio_data:
                await sio.emit('transcription_error', {'error': 'No audio data provided'}, room=sid)
                return
            
            # Convert base64 to proper format
            if isinstance(audio_data, str):
                audio_data = audio_data.split(',')[1] if ',' in audio_data else audio_data
            
            language_code = data.get('language', 'en-IN')
            
            result = await lightning_client.transcribe_audio(
                audio_data, 
                language_code, 
                48000,
                context_phrases=SPEECH_CONTEXT_PHRASES,
                context_boost=SPEECH_CONTEXT_BOOST
            )
            transcript = result.get('transcript', '')
            await sio.emit('transcription_result', {'text': transcript}, room=sid)
                
        except Exception as e:
            print(f"Transcription error: {e}")
            await sio.emit('transcription_error', {'error': str(e)}, room=sid)

    @sio.event
    async def agent_prompt(sid, data):
        """Handle agent prompts in different execution modes."""
        query, context, mode = data.get('prompt'), data.get('context', 'No context provided.'), data.get('mode', 'task')
        model = data.get('model', 'gemini')  # Get model selection from client
        language = data.get('language', 'en')  # Get language selection from client
        
        if not query:
            return
        
        session_manager.reset_cancellation(sid)
        await sio.emit('agent_working', {'working': True}, room=sid)
        
        try:
            if mode == 'iterative':
                await handle_iterative_mode(sid, query, context, sio, executor, session_manager, logger, model, language)
            elif mode == 'task':
                state = session_manager.get(sid)
                output_captures[sid] = state.output_capture
                cancellation_requests[sid] = state.cancelled
                await handle_task_mode(sid, query, context, sio, master_fd, output_captures, pending_plans, cancellation_requests, model, language)
                state.cancelled = cancellation_requests.get(sid, False)
            elif mode == 'ask':
                state = session_manager.get(sid)
                output_captures[sid] = state.output_capture
                cancellation_requests[sid] = state.cancelled
                await handle_ask_mode(sid, query, context, sio, master_fd, output_captures, cancellation_requests, model, language)
                state.cancelled = cancellation_requests.get(sid, False)
            elif mode == 'auto':
                state = session_manager.get(sid)
                output_captures[sid] = state.output_capture
                cancellation_requests[sid] = state.cancelled
                await handle_auto_mode(sid, query, context, sio, master_fd, output_captures, cancellation_requests, model, language)
                state.cancelled = cancellation_requests.get(sid, False)
        finally:
            if mode != 'task' or sid not in pending_plans:
                await sio.emit('agent_working', {'working': False}, room=sid)

    @sio.event
    async def user_approval(sid, data):
        """Handle user approval or rejection of execution plan in Task mode."""
        approved = data.get('approved', False)
        
        if sid not in pending_plans:
            await send_agent_message(sio, sid, "❌ No pending plan found.")
            await sio.emit('agent_working', {'working': False}, room=sid)
            return
        
        plan_data = pending_plans.pop(sid)
        plan, query, context = plan_data['plan'], plan_data['query'], plan_data['context']
        model = plan_data.get('model')
        language = plan_data.get('language', 'English')  # Already stored as full name
        
        await sio.emit('show_approval_buttons', {'show': False})
        
        if approved:
            await sio.emit('agent_working', {'working': True}, room=sid)
            await send_agent_message(sio, sid, "✅ Plan approved! Executing commands...")
            await asyncio.sleep(0.5)
            
            executed_commands = await execute_commands(sid, plan, sio, master_fd, output_captures, cancellation_requests)
            
            await sio.emit('agent_working', {'working': False}, room=sid)
            
            # Send completion status to chat
            status = "cancelled" if cancellation_requests.get(sid) else "completed"
            if status == "completed":
                await send_agent_message(sio, sid, f"✅ **Execution completed!** ({len(executed_commands)}/{len(plan)} commands executed)")
            else:
                await send_agent_message(sio, sid, f"🛑 **Execution cancelled.** ({len(executed_commands)}/{len(plan)} commands executed)")
            
            logger.log_interaction(
                session_id=sid, prompt=query, mode="task",
                plan=[{"command": cmd, "explanation": exp} for cmd, exp in plan],
                executed_commands=executed_commands,
                status=status,
                model=model,
                language=language
            )
            
            # Reset cancellation flag after execution completes
            cancellation_requests[sid] = False
        else:
            await send_agent_message(sio, sid, "❌ Plan rejected by user.")
            await sio.emit('agent_working', {'working': False}, room=sid)
            
            logger.log_interaction(
                session_id=sid, prompt=query, mode="task",
                plan=[{"command": cmd, "explanation": exp} for cmd, exp in plan],
                status="rejected",
                model=model,
                language=language
            )

    @sio.event
    async def cancel_agent(sid, data=None):
        """Handle cancellation request from user."""
        session_manager.cancel(sid)
        
        # Sync to legacy dicts for mode_handlers compatibility
        state = session_manager.get(sid)
        cancellation_requests[sid] = state.cancelled
        
        # Clear any pending plans
        if sid in pending_plans:
            pending_plans.pop(sid)
            # If there's a pending plan, send immediate feedback since no execution is happening
            await send_agent_message(sio, sid, "🛑 **Cancelled by user.**")
        
        # Ensure UI is reset
        await sio.emit('agent_working', {'working': False}, room=sid)
        await sio.emit('show_approval_buttons', {'show': False}, room=sid)


#  ITERATIVE MODE 

async def handle_iterative_mode(sid: str, query: str, context: str, sio, 
                                executor: CommandExecutor, 
                                session_manager: SessionManager, logger, model=None, language="en"):
    """Handle Iterative mode - execute commands adaptively without predetermined plan."""
    
    language_name = LANGUAGE_NAMES.get(language, 'English')
    
    # Initialize iterative session
    session_manager.get(sid).iterative_data = {
        'query': query, 'context': context, 'executed_commands': [],
        'active': True, 'step_number': 0
    }
    
    # Send mode initiation message
    await send_agent_message(sio, sid, 
        f"## 🔄 Iterative Mode\n\n**Goal:** {query}\n\n"
        "_The agent will execute commands step-by-step, adapting based on output from each step._")
    
    # Get first command decision
    first_decision = await generate_first_iterative_command(query, context, model=model, language=language)
    
    if await check_and_handle_cancellation(sio, sid, session_manager, logger, query, "iterative",
                                          plan=None,
                                          executed_commands=[], language=language_name):
        await sio.emit('agent_working', {'working': False}, room=sid)
        return
    
    first_command = first_decision.get('command')
    first_explanation = first_decision.get('explanation', '')
    
    if not first_command:
        await send_agent_message(sio, sid, f"❌ Agent could not determine first command.\n\n{first_explanation}")
        session_manager.get(sid).iterative_data = None
        await sio.emit('agent_working', {'working': False}, room=sid)
        logger.log_interaction(sid, query, "iterative", status="error", 
                             error=f"Could not determine first command: {first_explanation}", 
                             model=model, language=language_name)
        return
    
    # Execute first command
    session = session_manager.get(sid).iterative_data
    session['step_number'] = 1
    
    await send_agent_message(sio, sid, 
        f"### 🔄 Executing Step 1\n\n**Command:** `{first_command}`\n\n**Purpose:** {first_explanation}")
    
    if await check_and_handle_cancellation(sio, sid, session_manager, logger, query, "iterative",
                                          plan=None,
                                          executed_commands=[], language=language_name):
        await sio.emit('agent_working', {'working': False}, room=sid)
        return
    
    await executor.prepare_and_execute(sid, first_command)
    
    session['executed_commands'].append({'command': first_command, 'explanation': first_explanation})
    
    await executor.wait_for_completion(sid, timeout=120)
    
    if await check_and_handle_cancellation(sio, sid, session_manager, logger, query, "iterative",
                                          plan=None,
                                          executed_commands=session.get('executed_commands', []),
                                          language=language_name):
        await sio.emit('agent_working', {'working': False}, room=sid)
        return
    
    # Continue with iterative loop
    while session.get('active', False) and session['step_number'] < MAX_ITERATIONS:
        if await check_and_handle_cancellation(sio, sid, session_manager, logger, query, "iterative",
                                              plan=None,
                                              executed_commands=session.get('executed_commands', []),
                                              language=language_name):
            await sio.emit('agent_working', {'working': False}, room=sid)
            return
        
        # Get latest output
        latest_output = executor.session_manager.get(sid).output_capture.buffer
        
        if session['executed_commands']:
            session['executed_commands'][-1]['output'] = latest_output
        
        # Build command history
        previous_commands = "\n".join([f"{i+1}. {cmd['command']} - {cmd['explanation']}"
                                       for i, cmd in enumerate(session['executed_commands'])])
        
        # Get AI decision
        decision = await generate_iterative_decision(
            query, 
            previous_commands,
            latest_output[-2000:] if latest_output else "No output captured",
            model=model,
            language=language
        )
        
        if await check_and_handle_cancellation(sio, sid, session_manager, logger, query, "iterative",
                                              plan=None,
                                              executed_commands=session['executed_commands'],
                                              language=language_name):
            await sio.emit('agent_working', {'working': False}, room=sid)
            return
        
        # Check if we should continue
        if not decision.get('continue', False) or decision.get('status') == 'success':
            if decision.get('status') == 'success':
                await send_agent_message(sio, sid, f"✅ **Iterative execution completed successfully!**\n\n{decision.get('explanation', '')}")
            else:
                await send_agent_message(sio, sid, f"⚠️ Stopping execution.\n\n{decision.get('explanation', '')}")
            
            session['active'] = False
            executor.stop_capture(sid)
            await sio.emit('agent_working', {'working': False}, room=sid)
            
            logger.log_interaction(sid, query, "iterative",
                                 plan=None,
                                 executed_commands=session['executed_commands'],
                                 status="completed" if decision.get('status') == 'success' else "error",
                                 error=decision.get('explanation') if decision.get('status') != 'success' else None,
                                 model=model,
                                 language=language_name)
            session_manager.get(sid).iterative_data = None
            session_manager.reset_cancellation(sid)
            return
        
        # Execute next command
        if next_command := decision.get('next_command', ''):
            session['step_number'] += 1
            
            await send_agent_message(sio, sid, 
                f"### 🔄 Executing Step {session['step_number']}\n\n"
                f"**Command:** `{next_command}`\n\n"
                f"**Analysis:** {decision.get('explanation', '')}\n\n"
                f"_Adapting based on previous output..._"
            )
            
            if await check_and_handle_cancellation(sio, sid, session_manager, logger, query, "iterative",
                                                  plan=None,
                                                  executed_commands=session['executed_commands'],
                                                  language=language_name):
                await sio.emit('agent_working', {'working': False}, room=sid)
                return
            
            await executor.prepare_and_execute(sid, next_command)
            
            session['executed_commands'].append({'command': next_command, 'explanation': decision.get('explanation', '')})
            
            await executor.wait_for_completion(sid, timeout=300)
            
            if await check_and_handle_cancellation(sio, sid, session_manager, logger, query, "iterative",
                                                  plan=None,
                                                  executed_commands=session['executed_commands'],
                                                  language=language_name):
                await sio.emit('agent_working', {'working': False}, room=sid)
                return
        else:
            await send_agent_message(sio, sid, "❌ No next command provided.")
            session['active'] = False
            executor.stop_capture(sid)
            await sio.emit('agent_working', {'working': False}, room=sid)
            
            logger.log_interaction(sid, query, "iterative",
                                 plan=None,
                                 executed_commands=session['executed_commands'], status="error",
                                 error="No next command provided by AI",
                                 model=model,
                                 language=language_name)
            session_manager.get(sid).iterative_data = None
            session_manager.reset_cancellation(sid)
            return
    
    # Max iterations reached
    if session['step_number'] >= MAX_ITERATIONS:
        await send_agent_message(sio, sid, f"⚠️ Maximum iteration limit ({MAX_ITERATIONS}) reached. Stopping execution.")
        session['active'] = False
        executor.stop_capture(sid)
        await sio.emit('agent_working', {'working': False}, room=sid)
        
        logger.log_interaction(sid, query, "iterative",
                             plan=None,
                             executed_commands=session.get('executed_commands', []), status="error",
                             error=f"Maximum iteration limit ({MAX_ITERATIONS}) reached",
                             model=model,
                             language=language_name)
        session_manager.get(sid).iterative_data = None
        session_manager.reset_cancellation(sid)