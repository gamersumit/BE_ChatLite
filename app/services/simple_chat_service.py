"""
Simple chat service without OpenAI dependency for testing.
"""
import time
import random
from typing import List, Dict, Any, AsyncGenerator
import asyncio


class SimpleChatService:
    """Simple chat service that generates responses without external APIs."""
    
    def __init__(self):
        self.responses = [
            "Hello! I'm your AI assistant. How can I help you today?",
            "That's a great question! Let me help you with that.",
            "I understand what you're asking. Here's what I can tell you:",
            "Thanks for reaching out! I'd be happy to assist you.",
            "I'm here to help! Based on what you've asked:",
            "That's an interesting point. Let me provide some information:",
            "I appreciate your question. Here's my response:",
            "Great question! I can definitely help with that.",
        ]
        
        self.contextual_responses = {
            "hello": "Hello! Welcome to our website. How can I assist you today?",
            "hi": "Hi there! I'm your AI assistant. What can I help you with?",
            "help": "I'm here to help! You can ask me questions about our services, products, or general information.",
            "contact": "You can reach us through our contact page or call our support team directly.",
            "price": "For pricing information, I'd recommend checking our pricing page or contacting our sales team.",
            "support": "Our support team is available to help. You can reach them through the contact form or support email.",
            "product": "We offer various products and services. What specifically would you like to know about?",
            "service": "Our services are designed to meet your needs. What particular service interests you?",
            "about": "We're a company dedicated to providing excellent solutions. You can learn more on our about page.",
            "feature": "Our platform includes many useful features. What specific functionality are you looking for?",
        }
    
    async def generate_chat_response(
        self, 
        user_message: str,
        conversation_history: List[Dict[str, str]] = None,
        website_context: str = None,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """Generate a simple AI response."""
        start_time = time.time()
        
        try:
            # Simulate processing time
            await asyncio.sleep(0.5)
            
            # Generate response based on message content
            response = self._generate_response(user_message, conversation_history)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "response": response,
                "processing_time_ms": processing_time,
                "tokens_used": int(len(response.split()) * 1.3),  # Approximate token count
                "model_used": "simple-chat-v1",
                "confidence_score": random.uniform(0.8, 0.95),
                "cost_usd": 0.001  # Minimal cost for simple service
            }
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            return {
                "response": f"I encountered an issue: {str(e)}. Let me try to help you anyway!",
                "processing_time_ms": processing_time,
                "error": str(e),
                "tokens_used": 0,
                "model_used": "simple-chat-v1",
                "confidence_score": 0.5,
                "cost_usd": 0.0
            }
    
    async def stream_chat_response(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]] = None,
        website_context: str = None,
        max_tokens: int = 500
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a simple AI response word by word."""
        start_time = time.time()
        
        try:
            # Generate the full response first
            full_response = self._generate_response(user_message, conversation_history)
            words = full_response.split()
            
            # Stream word by word
            partial_response = ""
            for i, word in enumerate(words):
                partial_response += word + " "
                
                yield {
                    "type": "content",
                    "content": word + " ",
                    "partial": partial_response.strip(),
                    "is_complete": False
                }
                
                # Simulate typing speed
                await asyncio.sleep(0.1)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Send completion
            yield {
                "type": "complete",
                "content": full_response,
                "is_complete": True,
                "processing_time_ms": processing_time,
                "model_used": "simple-chat-v1"
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
    
    def _generate_response(self, user_message: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Generate a contextual response based on user message."""
        message_lower = user_message.lower()
        
        # Check for contextual keywords
        for keyword, response in self.contextual_responses.items():
            if keyword in message_lower:
                return f"{response}\n\nYou asked: '{user_message}'"
        
        # Check for questions
        if any(word in message_lower for word in ['what', 'how', 'when', 'where', 'why', 'which', 'who']):
            base_response = random.choice([
                "That's a great question! Based on your inquiry about",
                "I understand you're asking about",
                "Let me help you with your question regarding",
                "Good question! Regarding your inquiry about"
            ])
            return f"{base_response} '{user_message}', I'd be happy to provide more information. Could you be more specific about what you'd like to know?"
        
        # Check for greetings
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return random.choice([
                "Hello! Welcome to our website. I'm your AI assistant, ready to help you with any questions you might have.",
                "Hi there! Great to see you here. I'm here to assist you with information about our services and answer any questions.",
                "Hey! Thanks for visiting. I'm your helpful AI assistant. What can I help you with today?"
            ])
        
        # Check for thanks
        if any(word in message_lower for word in ['thank', 'thanks', 'appreciate']):
            return random.choice([
                "You're very welcome! I'm glad I could help. Is there anything else you'd like to know?",
                "My pleasure! Feel free to ask if you have any other questions.",
                "Happy to help! Let me know if you need assistance with anything else."
            ])
        
        # Default personalized response
        base_response = random.choice(self.responses)
        return f"{base_response}\n\nRegarding your message: '{user_message}', I understand you're looking for information. While I'd love to give you a more specific answer, I can help you find what you need. Could you tell me a bit more about what you're looking for?"