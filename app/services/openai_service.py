import time
from typing import List, Optional, Dict, Any, AsyncGenerator
import openai
from openai import AsyncOpenAI
from ..core.config import settings


class OpenAIService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        
    async def generate_chat_response(
        self, 
        user_message: str,
        conversation_history: List[Dict[str, str]] = None,
        website_context: str = None,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Generate AI response for chat message.
        
        Args:
            user_message: The user's message
            conversation_history: Previous messages in the conversation
            website_context: Relevant website content for RAG
            max_tokens: Maximum tokens in response
            
        Returns:
            Dict containing response and metadata
        """
        start_time = time.time()
        
        try:
            # Build conversation messages
            messages = self._build_messages(
                user_message, 
                conversation_history, 
                website_context
            )
            
            # Generate response using OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "response": response.choices[0].message.content,
                "processing_time_ms": processing_time,
                "tokens_used": response.usage.total_tokens,
                "model_used": self.model,
                "confidence_score": self._calculate_confidence(response),
                "cost_usd": self._calculate_cost(response.usage.total_tokens)
            }
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            print(f"❌ OpenAI service exception: {type(e).__name__}: {str(e)}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                "processing_time_ms": processing_time,
                "error": f"{type(e).__name__}: {str(e)}",
                "tokens_used": 0,
                "model_used": self.model,
                "confidence_score": 0.0,
                "cost_usd": 0.0
            }
    
    async def stream_chat_response(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]] = None,
        website_context: str = None,
        max_tokens: int = 500
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream AI response for real-time chat.
        
        Yields chunks of the response as they're generated.
        """
        start_time = time.time()
        
        try:
            messages = self._build_messages(
                user_message, 
                conversation_history, 
                website_context
            )
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                presence_penalty=0.1,
                frequency_penalty=0.1,
                stream=True
            )
            
            full_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    
                    yield {
                        "type": "chunk",
                        "content": content,
                        "full_response": full_response
                    }
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Final message with metadata
            yield {
                "type": "complete",
                "content": full_response,
                "processing_time_ms": processing_time,
                "model_used": self.model
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
    
    def _build_messages(
        self, 
        user_message: str, 
        conversation_history: List[Dict[str, str]] = None,
        website_context: str = None
    ) -> List[Dict[str, str]]:
        """Build messages array for OpenAI API."""
        messages = []
        
        # System prompt with website context
        system_prompt = self._build_system_prompt(website_context)
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-10:]:  # Limit to last 10 messages
                # Handle different conversation history formats
                if isinstance(msg, dict):
                    # Standard format with 'type' and 'content'
                    if "type" in msg and "content" in msg:
                        messages.append({
                            "role": "user" if msg["type"] == "user" else "assistant",
                            "content": msg["content"]
                        })
                    # Alternative format with 'message_type' and 'content'
                    elif "message_type" in msg and "content" in msg:
                        messages.append({
                            "role": "user" if msg["message_type"] == "user" else "assistant",
                            "content": msg["content"]
                        })
                    else:
                        print(f"⚠️ Unrecognized conversation history format: {list(msg.keys())}")
                else:
                    print(f"⚠️ Invalid conversation history item type: {type(msg)}")
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _build_system_prompt(self, website_context: str = None) -> str:
        """Build system prompt with website context."""
        base_prompt = """You are a helpful AI assistant embedded on a website to help visitors with their questions. 
        
        Guidelines:
        - Be friendly, professional, and concise
        - Answer questions based on the website content when available
        - If you don't know something specific to the website, be honest about it
        - Encourage users to contact the business directly for complex issues
        - Keep responses under 200 words unless more detail is specifically requested
        """
        
        if website_context:
            base_prompt += f"\n\nWebsite Content:\n{website_context}\n\nUse this information to answer questions about this specific website and business."
        
        return base_prompt
    
    def _calculate_confidence(self, response) -> float:
        """
        Calculate confidence score based on response characteristics.
        This is a simplified implementation - could be enhanced with more sophisticated analysis.
        """
        # For now, return a default confidence score
        # In a real implementation, you might analyze:
        # - Response length vs question complexity
        # - Presence of uncertain language
        # - Use of website context vs general knowledge
        return 0.85
    
    def _calculate_cost(self, total_tokens: int) -> float:
        """
        Calculate approximate cost for OpenAI API usage.
        Update these rates based on current OpenAI pricing.
        """
        # GPT-4o pricing (approximate, check current rates)
        cost_per_1k_tokens = 0.01  # $0.01 per 1K tokens (simplified)
        return (total_tokens / 1000) * cost_per_1k_tokens