"""
Message Importance Scoring Service for intelligent context optimization.
Scores messages based on relevance, information content, and conversation flow.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from collections import Counter

from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class MessageImportanceService:
    """
    Service for scoring message importance to optimize context selection.
    """
    
    # Keywords that indicate important messages
    IMPORTANT_KEYWORDS = {
        'question': ['what', 'how', 'why', 'when', 'where', 'which', 'who', '?'],
        'action': ['need', 'want', 'help', 'please', 'could', 'would', 'should', 'must'],
        'decision': ['yes', 'no', 'agree', 'confirm', 'choose', 'decide', 'select'],
        'problem': ['issue', 'problem', 'error', 'wrong', 'broken', 'fail', 'cannot', 'unable'],
        'information': ['price', 'cost', 'feature', 'specification', 'detail', 'information'],
        'critical': ['urgent', 'important', 'critical', 'asap', 'immediately', 'emergency']
    }
    
    # Message type weights
    MESSAGE_TYPE_WEIGHTS = {
        'user_question': 1.5,
        'user_clarification': 1.3,
        'user_confirmation': 1.1,
        'assistant_answer': 1.2,
        'assistant_information': 1.0,
        'assistant_confirmation': 0.8
    }
    
    def __init__(self):
        self.keyword_cache = {}
        self._compile_keyword_patterns()
    
    def _compile_keyword_patterns(self):
        """
        Compile regex patterns for efficient keyword matching.
        """
        for category, keywords in self.IMPORTANT_KEYWORDS.items():
            pattern = r'\b(' + '|'.join(re.escape(k) for k in keywords if k != '?') + r')\b'
            self.keyword_cache[category] = re.compile(pattern, re.IGNORECASE)
    
    def score_message(
        self,
        message: Dict[str, Any],
        conversation_context: Optional[List[Dict[str, Any]]] = None,
        position_in_conversation: Optional[int] = None
    ) -> float:
        """
        Score a message's importance on a scale of 0.0 to 2.0.
        
        Args:
            message: Message to score
            conversation_context: Optional surrounding messages for context
            position_in_conversation: Message position (affects recency score)
            
        Returns:
            Importance score (0.0-2.0, where 1.0 is average importance)
        """
        try:
            content = message.get('content', '')
            message_type = message.get('message_type', message.get('type', 'user'))
            
            # Base scores
            content_score = self._score_content(content)
            type_score = self._score_message_type(content, message_type)
            length_score = self._score_length(content)
            keyword_score = self._score_keywords(content)
            
            # Context-aware scores
            context_score = 1.0
            if conversation_context:
                context_score = self._score_in_context(message, conversation_context)
            
            # Position score (more recent = more important)
            position_score = 1.0
            if position_in_conversation is not None:
                position_score = self._score_position(position_in_conversation, len(conversation_context or []))
            
            # Combine scores with weights
            final_score = (
                content_score * 0.25 +
                type_score * 0.20 +
                length_score * 0.10 +
                keyword_score * 0.25 +
                context_score * 0.10 +
                position_score * 0.10
            )
            
            # Clamp to valid range
            final_score = max(0.0, min(2.0, final_score))
            
            logger.debug(f"📊 IMPORTANCE: Message scored {final_score:.2f} - "
                       f"content:{content_score:.2f}, type:{type_score:.2f}, "
                       f"keywords:{keyword_score:.2f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"❌ IMPORTANCE: Error scoring message: {e}")
            return 1.0  # Default to average importance on error
    
    def _score_content(self, content: str) -> float:
        """
        Score based on content characteristics.
        """
        score = 1.0
        
        # Questions are important
        if '?' in content:
            score += 0.3
        
        # Multiple sentences indicate more information
        sentence_count = len(re.split(r'[.!?]+', content))
        if sentence_count > 2:
            score += 0.2
        elif sentence_count == 1:
            score -= 0.1
        
        # Structured content (lists, numbered items) is important
        if any(marker in content for marker in ['\n-', '\n•', '\n1.', '\n*']):
            score += 0.3
        
        # URLs and references are important
        if re.search(r'https?://\S+', content):
            score += 0.2
        
        # Email addresses indicate contact information
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content):
            score += 0.3
        
        # Numbers (prices, quantities) are often important
        if re.search(r'\b\d+\.?\d*\b', content):
            score += 0.1
        
        return score
    
    def _score_message_type(self, content: str, message_type: str) -> float:
        """
        Score based on message type and content patterns.
        """
        content_lower = content.lower()
        
        # Determine specific message subtype
        if message_type == 'user':
            if '?' in content:
                return self.MESSAGE_TYPE_WEIGHTS['user_question']
            elif any(word in content_lower for word in ['clarify', 'mean', 'understand']):
                return self.MESSAGE_TYPE_WEIGHTS['user_clarification']
            elif any(word in content_lower for word in ['yes', 'no', 'confirm', 'agree']):
                return self.MESSAGE_TYPE_WEIGHTS['user_confirmation']
            else:
                return 1.2  # Default user message importance
        
        elif message_type == 'assistant':
            if len(content) > 200:  # Long responses usually contain information
                return self.MESSAGE_TYPE_WEIGHTS['assistant_information']
            elif any(word in content_lower for word in ['here', 'following', 'below']):
                return self.MESSAGE_TYPE_WEIGHTS['assistant_answer']
            elif any(word in content_lower for word in ['confirmed', 'noted', 'understood']):
                return self.MESSAGE_TYPE_WEIGHTS['assistant_confirmation']
            else:
                return 1.0  # Default assistant message importance
        
        return 1.0
    
    def _score_length(self, content: str) -> float:
        """
        Score based on message length (very short or very long messages may be less important).
        """
        length = len(content)
        
        if length < 20:  # Very short (like "ok", "thanks")
            return 0.7
        elif length < 50:
            return 0.9
        elif length < 500:
            return 1.0
        elif length < 1000:
            return 1.1
        else:  # Very long messages might be less focused
            return 0.95
    
    def _score_keywords(self, content: str) -> float:
        """
        Score based on presence of important keywords.
        """
        score = 1.0
        content_lower = content.lower()
        
        # Check each keyword category
        for category, pattern in self.keyword_cache.items():
            matches = len(pattern.findall(content_lower))
            
            if category == 'critical' and matches > 0:
                score += 0.5  # Critical keywords are very important
            elif category == 'problem' and matches > 0:
                score += 0.4
            elif category == 'question' and matches > 0:
                score += 0.3
            elif category == 'information' and matches > 0:
                score += 0.2
            elif category == 'action' and matches > 0:
                score += 0.2
            elif category == 'decision' and matches > 0:
                score += 0.3
        
        # Check for '?' separately (not in regex)
        if '?' in content:
            score += 0.2
        
        return min(2.0, score)  # Cap at 2.0
    
    def _score_in_context(
        self, 
        message: Dict[str, Any], 
        conversation_context: List[Dict[str, Any]]
    ) -> float:
        """
        Score message importance based on surrounding context.
        """
        score = 1.0
        
        # Find message position in context
        message_content = message.get('content', '')
        message_seq = message.get('sequence_number', message.get('sequence', -1))
        
        # Check if this message is referenced by later messages
        for other_msg in conversation_context:
            other_content = other_msg.get('content', '').lower()
            other_seq = other_msg.get('sequence_number', other_msg.get('sequence', -1))
            
            if other_seq > message_seq:
                # Check for references to this message
                if any(phrase in other_content for phrase in 
                       ['as mentioned', 'you said', 'earlier', 'above', 'previously']):
                    score += 0.3
                    break
        
        # Check if this message answers a previous question
        if message.get('message_type', message.get('type')) == 'assistant':
            # Look for preceding user question
            for i, other_msg in enumerate(conversation_context):
                if (other_msg.get('message_type', other_msg.get('type')) == 'user' and
                    '?' in other_msg.get('content', '') and
                    other_msg.get('sequence_number', other_msg.get('sequence', -1)) < message_seq):
                    score += 0.2
                    break
        
        return score
    
    def _score_position(self, position: int, total_messages: int) -> float:
        """
        Score based on position in conversation (recency bias).
        """
        if total_messages == 0:
            return 1.0
        
        # Normalize position (0 = oldest, 1 = newest)
        normalized_position = position / total_messages
        
        # Recent messages are more important (exponential decay)
        recency_score = 0.7 + (0.6 * normalized_position ** 2)
        
        # But the very first messages establish context
        if position < 3:
            recency_score += 0.2
        
        return recency_score
    
    def score_conversation(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Score all messages in a conversation.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            List of tuples (message, score) sorted by importance
        """
        scored_messages = []
        
        for i, message in enumerate(messages):
            score = self.score_message(
                message,
                conversation_context=messages,
                position_in_conversation=i
            )
            scored_messages.append((message, score))
        
        # Sort by score (highest first)
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        
        return scored_messages
    
    def select_important_messages(
        self,
        messages: List[Dict[str, Any]],
        max_messages: int,
        min_importance_score: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Select the most important messages from a conversation.
        
        Args:
            messages: List of conversation messages
            max_messages: Maximum number of messages to select
            min_importance_score: Minimum importance score threshold
            
        Returns:
            List of selected messages in conversation order
        """
        # Score all messages
        scored_messages = self.score_conversation(messages)
        
        # Filter by minimum score and limit
        selected = []
        for message, score in scored_messages:
            if score >= min_importance_score and len(selected) < max_messages:
                selected.append(message)
        
        # If we don't have enough messages meeting the threshold, add top scored ones
        if len(selected) < max_messages:
            for message, score in scored_messages:
                if message not in selected:
                    selected.append(message)
                    if len(selected) >= max_messages:
                        break
        
        # Sort back to conversation order
        selected.sort(key=lambda m: m.get('sequence_number', m.get('sequence', 0)))
        
        return selected
    
    def get_importance_statistics(
        self,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get statistics about message importance in a conversation.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Dictionary of statistics
        """
        if not messages:
            return {
                'avg_importance': 0.0,
                'max_importance': 0.0,
                'min_importance': 0.0,
                'high_importance_count': 0,
                'total_messages': 0
            }
        
        scores = [self.score_message(msg, messages, i) for i, msg in enumerate(messages)]
        
        return {
            'avg_importance': sum(scores) / len(scores),
            'max_importance': max(scores),
            'min_importance': min(scores),
            'high_importance_count': sum(1 for s in scores if s >= 1.3),
            'total_messages': len(messages),
            'importance_distribution': {
                'very_low': sum(1 for s in scores if s < 0.5),
                'low': sum(1 for s in scores if 0.5 <= s < 0.8),
                'average': sum(1 for s in scores if 0.8 <= s < 1.2),
                'high': sum(1 for s in scores if 1.2 <= s < 1.5),
                'very_high': sum(1 for s in scores if s >= 1.5)
            }
        }