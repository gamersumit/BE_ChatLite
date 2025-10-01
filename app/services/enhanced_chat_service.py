"""
Enhanced chat service that uses OpenAI with fallback to simple chat service.
"""
import time
from typing import List, Optional, Dict, Any, AsyncGenerator
import asyncio
from ..core.config import settings
from .openai_service import OpenAIService
from .simple_chat_service import SimpleChatService


class EnhancedChatService:
    """Enhanced chat service with OpenAI integration and fallback."""
    
    def __init__(self):
        self.openai_service = OpenAIService()
        self.simple_service = SimpleChatService()
        self.use_openai = bool(settings.openai_api_key)
        
    async def generate_chat_response(
        self, 
        user_message: str,
        conversation_history: List[Dict[str, str]] = None,
        website_context: str = None,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Generate AI response using OpenAI with fallback to simple service.
        """
        if self.use_openai:
            try:
                # Try OpenAI first
                response = await self.openai_service.generate_chat_response(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    website_context=website_context,
                    max_tokens=max_tokens
                )
                
                # If OpenAI returned an error response, try simple service
                if response.get("error") is not None or "technical difficulties" in response.get("response", ""):
                    error_msg = response.get("error", "No specific error")
                    print(f"⚠️ OpenAI returned error: {error_msg}, falling back to simple service")
                    return await self.simple_service.generate_chat_response(
                        user_message=user_message,
                        conversation_history=conversation_history,
                        website_context=website_context,
                        max_tokens=max_tokens
                    )
                
                return response
                
            except Exception as e:
                print(f"⚠️ OpenAI service failed with exception: {type(e).__name__}: {str(e)}, falling back to simple service")
                return await self.simple_service.generate_chat_response(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    website_context=website_context,
                    max_tokens=max_tokens
                )
        else:
            # Use simple service directly if no OpenAI key
            return await self.simple_service.generate_chat_response(
                user_message=user_message,
                conversation_history=conversation_history,
                website_context=website_context,
                max_tokens=max_tokens
            )
    
    async def stream_chat_response(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]] = None,
        website_context: str = None,
        max_tokens: int = 500
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream AI response using OpenAI with fallback to simple service.
        """
        if self.use_openai:
            try:
                # Try OpenAI streaming first
                has_content = False
                async for chunk in self.openai_service.stream_chat_response(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    website_context=website_context,
                    max_tokens=max_tokens
                ):
                    if chunk.get("type") == "chunk" and chunk.get("content"):
                        has_content = True
                    elif chunk.get("type") == "error":
                        # If OpenAI streaming fails, fall back to simple service
                        print("⚠️ OpenAI streaming failed, falling back to simple service")
                        async for fallback_chunk in self.simple_service.stream_chat_response(
                            user_message=user_message,
                            conversation_history=conversation_history,
                            website_context=website_context,
                            max_tokens=max_tokens
                        ):
                            yield fallback_chunk
                        return
                    yield chunk
                    
                # If no content was streamed, fall back
                if not has_content:
                    print("⚠️ No content from OpenAI, falling back to simple service")
                    async for fallback_chunk in self.simple_service.stream_chat_response(
                        user_message=user_message,
                        conversation_history=conversation_history,
                        website_context=website_context,
                        max_tokens=max_tokens
                    ):
                        yield fallback_chunk
                        
            except Exception as e:
                print(f"⚠️ OpenAI streaming service failed: {e}, falling back to simple service")
                async for chunk in self.simple_service.stream_chat_response(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    website_context=website_context,
                    max_tokens=max_tokens
                ):
                    yield chunk
        else:
            # Use simple service directly if no OpenAI key
            async for chunk in self.simple_service.stream_chat_response(
                user_message=user_message,
                conversation_history=conversation_history,
                website_context=website_context,
                max_tokens=max_tokens
            ):
                yield chunk
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of available services."""
        return {
            "openai_available": self.use_openai,
            "simple_service_available": True,
            "current_primary": "openai" if self.use_openai else "simple",
            "model": settings.openai_model if self.use_openai else "simple-chat-v1"
        }