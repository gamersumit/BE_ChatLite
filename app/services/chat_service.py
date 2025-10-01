from typing import List, Optional, Dict, Any
from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from ..models import Conversation, Message, Website
from ..schemas.chat import ChatMessage, ChatResponse, ChatHistory
from .enhanced_chat_service import EnhancedChatService


class ChatService:
    def __init__(self, enhanced_chat_service: EnhancedChatService):
        self.chat_service = enhanced_chat_service
    
    async def process_chat_message(
        self, 
        chat_request: ChatMessage, 
        db: AsyncSession
    ) -> ChatResponse:
        """
        Process incoming chat message and generate AI response.
        """
        # Get or create conversation
        print("chat_request")
        conversation = await self._get_or_create_conversation(
            chat_request.session_id,
            chat_request.page_url,
            chat_request.page_title,
            chat_request.user_id,
            db
        )
        
        # Save user message
        print("conversation")
        user_message = await self._save_user_message(
            conversation.id,
            chat_request.message,
            db
        )
        
        # Get conversation history
        print("history")
        history = await self._get_conversation_history(conversation.id, db)
        
        # Get website context (simplified for MVP)
        print("website_context")
        website_context = await self._get_website_context(conversation.website_id, db)
        
        # Generate AI response using simple chat service
        print("ai_response")
        ai_response = await self.chat_service.generate_chat_response(
            user_message=chat_request.message,
            conversation_history=history,
            website_context=website_context
        )
        
        # Save AI response
        print("ai_message")
        ai_message = await self._save_ai_message(
            conversation.id,
            ai_response,
            db
        )
        
        # Update conversation metrics
        print("update_conversation_metrics")
        await self._update_conversation_metrics(conversation.id, db)
        
        return ChatResponse(
            message_id=ai_message.id,
            response=ai_response["response"],
            confidence_score=ai_response.get("confidence_score"),
            processing_time_ms=ai_response["processing_time_ms"],
            sources=ai_response.get("sources", [])
        )
    
    async def get_chat_history(
        self, 
        session_id: str, 
        db: AsyncSession
    ) -> Optional[ChatHistory]:
        """
        Retrieve chat history for a session.
        """
        # Get conversation
        result = await db.execute(
            select(Conversation).where(Conversation.session_id == session_id)
        )
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            return None
        
        # Get messages
        messages_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.sequence_number)
        )
        messages = messages_result.scalars().all()
        
        return ChatHistory(
            conversation_id=conversation.id,
            session_id=conversation.session_id,
            messages=messages,
            total_messages=len(messages),
            started_at=conversation.started_at,
            last_activity_at=conversation.last_activity_at
        )
    
    async def _get_or_create_conversation(
        self,
        session_id: str,
        page_url: Optional[str],
        page_title: Optional[str],
        user_id: Optional[str],
        db: AsyncSession
    ) -> Conversation:
        """Get existing conversation or create new one."""
        
        # Try to find existing active conversation
        result = await db.execute(
            select(Conversation).where(
                and_(
                    Conversation.session_id == session_id,
                    Conversation.is_active == True
                )
            )
        )
        conversation = result.scalar_one_or_none()
        
        if conversation:
            # Update last activity
            conversation.last_activity_at = datetime.now(timezone.utc)
            if page_url:
                conversation.page_url = page_url
            if page_title:
                conversation.page_title = page_title
            await db.commit()
            return conversation
        
        # Create new conversation (simplified - assumes website exists)
        # In production, you'd validate widget_id and get website_id
        website_result = await db.execute(select(Website).limit(1))
        website = website_result.scalar_one_or_none()
        
        if not website:
            # Create a default website for MVP
            website = Website(
                name="Default Website",
                url="https://example.com",
                domain="example.com",
                widget_id="default_widget"
            )
            db.add(website)
            await db.flush()
        
        conversation = Conversation(
            session_id=session_id,
            website_id=website.id,
            user_id=user_id,
            page_url=page_url,
            page_title=page_title,
            is_active=True
        )
        
        db.add(conversation)
        await db.commit()
        
        return conversation
    
    async def _save_user_message(
        self,
        conversation_id,
        content: str,
        db: AsyncSession
    ) -> Message:
        """Save user message to database."""
        
        # Get next sequence number
        result = await db.execute(
            select(func.coalesce(func.max(Message.sequence_number), 0))
            .where(Message.conversation_id == conversation_id)
        )
        max_sequence = result.scalar() or 0
        
        message = Message(
            conversation_id=conversation_id,
            content=content,
            message_type="user",
            sequence_number=max_sequence + 1,
            word_count=len(content.split()),
            character_count=len(content)
        )
        
        db.add(message)
        await db.commit()
        await db.refresh(message)
        
        return message
    
    async def _save_ai_message(
        self,
        conversation_id,
        ai_response: Dict[str, Any],
        db: AsyncSession
    ) -> Message:
        """Save AI response message to database."""
        
        # Get next sequence number
        result = await db.execute(
            select(func.coalesce(func.max(Message.sequence_number), 0))
            .where(Message.conversation_id == conversation_id)
        )
        max_sequence = result.scalar() or 0
        
        message = Message(
            conversation_id=conversation_id,
            content=ai_response["response"],
            message_type="assistant",
            sequence_number=max_sequence + 1,
            word_count=len(ai_response["response"].split()),
            character_count=len(ai_response["response"]),
            processing_time_ms=ai_response["processing_time_ms"],
            model_used=ai_response.get("model_used"),
            tokens_used=ai_response.get("tokens_used"),
            cost_usd=ai_response.get("cost_usd"),
            confidence_score=ai_response.get("confidence_score")
        )
        
        db.add(message)
        await db.commit()
        await db.refresh(message)
        
        return message
    
    async def _get_conversation_history(
        self,
        conversation_id,
        db: AsyncSession,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """Get recent conversation history."""
        
        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.sequence_number.desc())
            .limit(limit)
        )
        messages = result.scalars().all()
        
        # Reverse to get chronological order
        history = []
        for message in reversed(messages):
            history.append({
                "type": message.message_type,
                "content": message.content
            })
        
        return history
    
    async def _get_website_context(
        self,
        website_id,
        db: AsyncSession
    ) -> Optional[str]:
        """
        Get website context for RAG.
        Simplified for MVP - in production, this would query scraped content.
        """
        result = await db.execute(
            select(Website).where(Website.id == website_id)
        )
        website = result.scalar_one_or_none()
        
        if not website:
            return None
        
        # Build basic context from website info
        context_parts = [
            f"Business Name: {website.business_name}" if website.business_name else "",
            f"Website: {website.url}",
            f"Business Description: {website.business_description}" if website.business_description else "",
            f"Contact Email: {website.contact_email}" if website.contact_email else ""
        ]
        
        return "\n".join(filter(None, context_parts))
    
    async def _update_conversation_metrics(
        self,
        conversation_id,
        db: AsyncSession
    ):
        """Update conversation message counts."""
        
        # Count messages by type
        result = await db.execute(
            select(
                func.count(Message.id).label('total'),
                func.sum(func.case((Message.message_type == 'user', 1), else_=0)).label('user'),
                func.sum(func.case((Message.message_type == 'assistant', 1), else_=0)).label('ai')
            )
            .where(Message.conversation_id == conversation_id)
        )
        counts = result.first()
        
        # Update conversation
        conversation_result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = conversation_result.scalar_one()
        
        conversation.total_messages = counts.total or 0
        conversation.user_messages = counts.user or 0
        conversation.ai_messages = counts.ai or 0
        conversation.last_activity_at = datetime.now(timezone.utc)
        
        await db.commit()