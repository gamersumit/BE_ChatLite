"""
Socket-based chat service - Basic implementation.
"""
import socket
import threading
import json
from typing import Dict, Any, Optional

# Global socket server instance
_socket_server: Optional["SocketChatService"] = None


class SocketChatService:
    """Basic socket-based chat server."""
    
    def __init__(self, host='127.0.0.1', port=8002):
        self.host = host
        self.port = port
        self.clients = {}
        self.running = False
        
    def start(self):
        """Start the socket server."""
        self.running = True
        print(f"Socket server started on {self.host}:{self.port}")
        
    def stop(self):
        """Stop the socket server."""
        self.running = False
        print("Socket server stopped")
        
    def broadcast_message(self, message: str, room_id: str = None):
        """Broadcast message to clients."""
        print(f"Broadcasting message: {message}")
        
    def handle_message(self, client_socket, data: Dict[str, Any]):
        """Handle incoming message."""
        return {"status": "received", "message": "Message handled"}


def get_socket_server() -> SocketChatService:
    """Get or create socket server instance."""
    global _socket_server
    if _socket_server is None:
        _socket_server = SocketChatService()
    return _socket_server


def start_socket_server():
    """Start the socket server."""
    server = get_socket_server()
    server.start()
    
    
def stop_socket_server():
    """Stop the socket server."""
    global _socket_server
    if _socket_server:
        _socket_server.stop()
        _socket_server = None