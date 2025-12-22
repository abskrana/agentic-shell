"""Messaging utilities for Socket.IO communication.

Helper function for sending agent messages to connected clients.
All messages from the agent are sent as 'agent_message' events.
"""


async def send_agent_message(sio, sid: str, message: str) -> None:
    """Send an agent message to the chat panel.
    
    All non-user messages should use this function, including system notifications,
    error messages, and regular agent responses.
    
    Args:
        sio: Socket.IO server instance.
        sid: Session ID of the target client.
        message: Message content to send.
    """
    await sio.emit('agent_message', {'message': message}, room=sid)
