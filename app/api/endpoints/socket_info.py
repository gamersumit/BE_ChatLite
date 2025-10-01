"""
Socket server information endpoint.
"""
from fastapi import APIRouter, HTTPException
from ...services.socket_chat_service import get_socket_server

router = APIRouter()


@router.get("/info")
async def get_socket_info():
    """Get socket server connection information."""
    socket_server = get_socket_server()
    
    if socket_server is None:
        raise HTTPException(status_code=503, detail="Socket server not running")
    
    return {
        "status": "running",
        "host": socket_server.host,
        "port": socket_server.port,
        "active_sessions": len(socket_server.clients),
        "session_ids": list(socket_server.clients.keys()),
        "connection_url": f"{socket_server.host}:{socket_server.port}",
        "message_format": {
            "chat": {
                "type": "chat",
                "session_id": "string",
                "message": "string",
                "page_url": "string (optional)",
                "page_title": "string (optional)"
            },
            "ping": {
                "type": "ping", 
                "session_id": "string"
            }
        },
        "response_types": [
            "chat_response",
            "pong", 
            "error"
        ]
    }


@router.get("/sessions")
async def get_active_sessions():
    """Get list of active socket sessions."""
    socket_server = get_socket_server()
    
    if socket_server is None:
        raise HTTPException(status_code=503, detail="Socket server not running")
    
    return {
        "active_sessions": socket_server.get_active_sessions(),
        "total_count": len(socket_server.clients)
    }


@router.post("/broadcast")
async def broadcast_message(message: dict):
    """Broadcast a message to all active sessions."""
    socket_server = get_socket_server()
    
    if socket_server is None:
        raise HTTPException(status_code=503, detail="Socket server not running")
    
    socket_server.broadcast_to_all(message)
    
    return {
        "status": "broadcast_sent",
        "message": message,
        "recipients": len(socket_server.clients)
    }