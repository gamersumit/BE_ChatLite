from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from uuid import UUID


class ChatMessage(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., min_length=1, max_length=2000, description="User message content")
    page_url: Optional[str] = Field(None, description="Current page URL")
    page_title: Optional[str] = Field(None, description="Current page title")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_123456789",
                "message": "How can I contact customer support?",
                "page_url": "https://example.com/contact",
                "page_title": "Contact Us - Example Company"
            }
        }


class ChatResponse(BaseModel):
    message_id: UUID = Field(..., description="Unique message identifier")
    response: str = Field(..., description="AI assistant response")
    confidence_score: Optional[float] = Field(None, description="Response confidence score")
    processing_time_ms: int = Field(..., description="Response generation time")
    sources: Optional[List[str]] = Field(None, description="Content sources used for response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "550e8400-e29b-41d4-a716-446655440000",
                "response": "You can contact our customer support team through...",
                "confidence_score": 0.95,
                "processing_time_ms": 1500,
                "sources": ["Contact Page", "FAQ Section"]
            }
        }


class MessageHistory(BaseModel):
    id: UUID
    content: str
    message_type: str  # "user" or "assistant"
    sequence_number: int
    sent_at: datetime
    was_helpful: Optional[bool] = None
    
    class Config:
        from_attributes = True


class ChatHistory(BaseModel):
    conversation_id: UUID
    session_id: str
    messages: List[MessageHistory]
    total_messages: int
    started_at: datetime
    last_activity_at: datetime
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "session_id": "sess_123456789",
                "total_messages": 4,
                "started_at": "2025-08-20T10:30:00Z",
                "last_activity_at": "2025-08-20T10:35:00Z",
                "messages": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440001",
                        "content": "Hello, how can I help you?",
                        "message_type": "assistant",
                        "sequence_number": 1,
                        "sent_at": "2025-08-20T10:30:00Z"
                    }
                ]
            }
        }


class TypingIndicator(BaseModel):
    session_id: str
    is_typing: bool = Field(default=True)
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_123456789",
                "is_typing": True
            }
        }