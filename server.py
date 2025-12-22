"""
Agentic Shell
An AI-powered terminal assistant with multi-language support and adaptive execution modes.
Powered by Lightning AI for scalable AI inference and speech-to-text capabilities.
"""

import os
import sys
import asyncio
import pty
import subprocess

from aiohttp import web
import socketio
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from config import HOST, PORT, lightning as lightning_config
from socket_handlers import register_handlers
from ai_brain import configure_model
from lightning_client import LightningAIModel
from utils.logger import initialize_logger

# Initialize logger
logger = initialize_logger(
    lightning_api_url=f"{lightning_config.unified_backend_url}/api/logs"
)

# Validate Lightning AI configuration
if not lightning_config.unified_backend_url:
    print("ERROR: Lightning AI Backend Not Configured")
    print("The LIGHTNING_UNIFIED_URL environment variable is required.")
    print("Please create a .env file with your Lightning AI URL:")
    print()
    print("  LIGHTNING_UNIFIED_URL=https://your-app.lightning.ai")
    print()
    sys.exit(1)

# Initialize Lightning AI model
model = LightningAIModel(base_url=lightning_config.unified_backend_url)

# Make Lightning client available to socket handlers
import socket_handlers
socket_handlers.lightning_client = model.client

# Configure AI brain with the model
configure_model(model)

# Initialize Socket.IO server with CORS support
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application()
sio.attach(app)

# Create pseudo-terminal for interactive bash session
master_fd, slave_fd = pty.openpty()

env = os.environ.copy()
env['TERM'] = 'xterm-256color'

# Start bash process in the PTY
subprocess.Popen(
    ['bash', '-i'],
    stdin=slave_fd,
    stdout=slave_fd,
    stderr=slave_fd,
    env=env,
    preexec_fn=os.setsid
)

# Session management storage
session_storage = {
    'output_captures': {},
    'pending_plans': {},
    'iterative_sessions': {},
    'cancellation_requests': {}
}

# Register Socket.IO event handlers
register_handlers(
    sio,
    master_fd,
    session_storage['output_captures'],
    session_storage['pending_plans'],
    session_storage['iterative_sessions'],
    session_storage['cancellation_requests']
)



async def pty_output_forwarder():
    """Forward PTY output to connected clients in real-time.
    
    Continuously reads from the master PTY file descriptor and broadcasts
    output to all connected Socket.IO clients while updating capture buffers.
    """
    loop = asyncio.get_running_loop()
    
    def on_pty_output():
        """Callback invoked when data is available from the PTY."""
        try:
            data = os.read(master_fd, 1024)
            if data:
                decoded_data = data.decode(errors='ignore')
                
                # Update capture buffers for all active sessions
                for capture in session_storage['output_captures'].values():
                    capture.append(decoded_data)
                
                # Broadcast to all connected clients
                asyncio.create_task(
                    sio.emit('pty_output', {'output': decoded_data})
                )
        except OSError:
            pass
    
    loop.add_reader(master_fd, on_pty_output)


# Configure HTTP routes
app.router.add_get('/', lambda request: web.FileResponse('./static/index.html'))
app.router.add_static('/static', path='./static')

# Start PTY output forwarder on application startup
app.on_startup.append(lambda app: asyncio.create_task(pty_output_forwarder()))


def main():
    """Main entry point for the application."""
    print()
    print("Agentic Shell")
    print(f"Ctrl+Click this link: http://localhost:{PORT}")
    print(f"Voice Support: 12 languages")
    print(f"Execution Modes: Task | Ask | Auto | Iterative")
    print()
    web.run_app(app, host=HOST, port=PORT)        


if __name__ == '__main__':
    main()
