"""
Script Generation Service for dynamic JavaScript widget script creation.
Handles template engine, configuration injection, versioning, and CDN preparation.
"""

import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from ..models.website import Website
from ..models.script_version import ScriptVersion


class ScriptGenerationService:
    """Service for generating and managing widget installation scripts."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.script_template = self._load_script_template()
        self.cdn_base_url = os.getenv("CDN_BASE_URL", "https://cdn.chatlite.app")
    
    def _load_script_template(self) -> str:
        """Load the JavaScript template for widget generation."""
        return """
(function() {
    'use strict';
    
    // Widget configuration injected at build time
    const WIDGET_CONFIG = /*CONFIG_PLACEHOLDER*/;
    
    // Security: Validate domain
    const allowedDomains = WIDGET_CONFIG.allowedDomains || [];
    const currentDomain = window.location.hostname;
    
    if (allowedDomains.length > 0 && !allowedDomains.includes(currentDomain)) {
        console.warn('ChatLite widget not authorized for domain:', currentDomain);
        return;
    }
    
    // Widget state
    let widget = null;
    let isOpen = false;
    let isInitialized = false;
    
    // Widget HTML template
    const WIDGET_HTML = `
        <div id="chatlite-widget-${WIDGET_CONFIG.widgetId}" class="chatlite-widget" style="display: none;">
            <div class="chatlite-widget-button" id="chatlite-toggle-${WIDGET_CONFIG.widgetId}">
                ${WIDGET_CONFIG.showAvatar ? '<div class="chatlite-avatar"></div>' : ''}
                <div class="chatlite-chat-icon">💬</div>
                ${WIDGET_CONFIG.showOnlineStatus ? '<div class="chatlite-online-indicator"></div>' : ''}
            </div>
            <div class="chatlite-chat-window" id="chatlite-window-${WIDGET_CONFIG.widgetId}">
                <div class="chatlite-chat-header">
                    <div class="chatlite-header-content">
                        ${WIDGET_CONFIG.customLogoUrl ? 
                            `<img src="${WIDGET_CONFIG.customLogoUrl}" alt="Logo" class="chatlite-logo">` : ''}
                        <div class="chatlite-header-text">
                            <h3>${WIDGET_CONFIG.companyName || 'Chat Support'}</h3>
                            ${WIDGET_CONFIG.showOnlineStatus ? '<span class="chatlite-status">Online</span>' : ''}
                        </div>
                    </div>
                    <button class="chatlite-close-btn" id="chatlite-close-${WIDGET_CONFIG.widgetId}">×</button>
                </div>
                <div class="chatlite-chat-messages" id="chatlite-messages-${WIDGET_CONFIG.widgetId}">
                    ${WIDGET_CONFIG.welcomeMessage ? 
                        `<div class="chatlite-message chatlite-bot-message">
                            <div class="chatlite-message-content">${WIDGET_CONFIG.welcomeMessage}</div>
                        </div>` : ''}
                </div>
                <div class="chatlite-chat-input-container">
                    <input type="text" 
                           id="chatlite-input-${WIDGET_CONFIG.widgetId}" 
                           placeholder="${WIDGET_CONFIG.placeholderText || 'Type your message...'}" 
                           class="chatlite-chat-input">
                    <button id="chatlite-send-${WIDGET_CONFIG.widgetId}" class="chatlite-send-btn">Send</button>
                </div>
                ${WIDGET_CONFIG.showBranding ? 
                    '<div class="chatlite-branding">Powered by ChatLite</div>' : ''}
            </div>
        </div>
    `;
    
    // Dynamic CSS generation based on configuration
    function generateCSS() {
        const position = WIDGET_CONFIG.widgetPosition || 'bottom-right';
        const color = WIDGET_CONFIG.widgetColor || '#0066CC';
        const size = WIDGET_CONFIG.widgetSize || 'medium';
        const theme = WIDGET_CONFIG.widgetTheme || 'light';
        const borderRadius = WIDGET_CONFIG.borderRadius || 8;
        const fontFamily = WIDGET_CONFIG.fontFamily || 'system-ui, -apple-system, sans-serif';
        
        const sizeMap = {
            small: { button: '50px', window: '300px' },
            medium: { button: '60px', window: '350px' },
            large: { button: '70px', window: '400px' }
        };
        
        const positionMap = {
            'bottom-right': { bottom: '20px', right: '20px' },
            'bottom-left': { bottom: '20px', left: '20px' },
            'top-right': { top: '20px', right: '20px' },
            'top-left': { top: '20px', left: '20px' }
        };
        
        const themeColors = theme === 'dark' ? {
            bg: '#2d3748',
            text: '#ffffff',
            border: '#4a5568',
            inputBg: '#4a5568'
        } : {
            bg: '#ffffff',
            text: '#333333',
            border: '#e2e8f0',
            inputBg: '#f7fafc'
        };
        
        return `
            .chatlite-widget {
                position: fixed;
                ${Object.entries(positionMap[position]).map(([k, v]) => `${k}: ${v}`).join('; ')};
                z-index: 999999;
                font-family: ${fontFamily};
                font-size: 14px;
                line-height: 1.4;
            }
            
            .chatlite-widget-button {
                width: ${sizeMap[size].button};
                height: ${sizeMap[size].button};
                background: ${color};
                border-radius: 50%;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                transition: all 0.3s ease;
                position: relative;
            }
            
            .chatlite-widget-button:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
            }
            
            .chatlite-chat-icon {
                font-size: ${size === 'large' ? '28px' : size === 'medium' ? '24px' : '20px'};
                color: white;
            }
            
            .chatlite-online-indicator {
                position: absolute;
                top: -2px;
                right: -2px;
                width: 12px;
                height: 12px;
                background: #10b981;
                border: 2px solid white;
                border-radius: 50%;
            }
            
            .chatlite-chat-window {
                position: absolute;
                bottom: ${sizeMap[size].button === '50px' ? '60px' : sizeMap[size].button === '60px' ? '70px' : '80px'};
                right: 0;
                width: ${sizeMap[size].window};
                height: 400px;
                background: ${themeColors.bg};
                border: 1px solid ${themeColors.border};
                border-radius: ${borderRadius}px;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
                display: none;
                flex-direction: column;
                overflow: hidden;
            }
            
            .chatlite-chat-header {
                background: ${color};
                color: white;
                padding: 12px 16px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            
            .chatlite-header-content {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .chatlite-logo {
                width: 24px;
                height: 24px;
                border-radius: 4px;
            }
            
            .chatlite-header-text h3 {
                margin: 0;
                font-size: 14px;
                font-weight: 600;
            }
            
            .chatlite-status {
                font-size: 11px;
                opacity: 0.9;
            }
            
            .chatlite-close-btn {
                background: none;
                border: none;
                color: white;
                font-size: 20px;
                cursor: pointer;
                padding: 0;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .chatlite-chat-messages {
                flex: 1;
                padding: 16px;
                overflow-y: auto;
                background: ${themeColors.bg};
            }
            
            .chatlite-message {
                margin-bottom: 12px;
            }
            
            .chatlite-message-content {
                padding: 8px 12px;
                border-radius: 12px;
                max-width: 80%;
                word-wrap: break-word;
            }
            
            .chatlite-bot-message .chatlite-message-content {
                background: ${themeColors.border};
                color: ${themeColors.text};
            }
            
            .chatlite-user-message {
                text-align: right;
            }
            
            .chatlite-user-message .chatlite-message-content {
                background: ${color};
                color: white;
                margin-left: auto;
            }
            
            .chatlite-chat-input-container {
                padding: 12px 16px;
                border-top: 1px solid ${themeColors.border};
                display: flex;
                gap: 8px;
                background: ${themeColors.bg};
            }
            
            .chatlite-chat-input {
                flex: 1;
                padding: 8px 12px;
                border: 1px solid ${themeColors.border};
                border-radius: 20px;
                background: ${themeColors.inputBg};
                color: ${themeColors.text};
                outline: none;
                font-size: 14px;
            }
            
            .chatlite-chat-input:focus {
                border-color: ${color};
            }
            
            .chatlite-send-btn {
                padding: 8px 16px;
                background: ${color};
                color: white;
                border: none;
                border-radius: 20px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
            }
            
            .chatlite-send-btn:hover {
                opacity: 0.9;
            }
            
            .chatlite-branding {
                padding: 8px 16px;
                font-size: 11px;
                color: #9ca3af;
                text-align: center;
                border-top: 1px solid ${themeColors.border};
                background: ${themeColors.bg};
            }
            
            ${WIDGET_CONFIG.customCSS || ''}
        `;
    }
    
    // Initialize widget
    function initWidget() {
        if (isInitialized) return;
        
        // Inject CSS
        const style = document.createElement('style');
        style.textContent = generateCSS();
        document.head.appendChild(style);
        
        // Inject HTML
        const widgetContainer = document.createElement('div');
        widgetContainer.innerHTML = WIDGET_HTML;
        document.body.appendChild(widgetContainer.firstElementChild);
        
        widget = document.getElementById(`chatlite-widget-${WIDGET_CONFIG.widgetId}`);
        const toggleBtn = document.getElementById(`chatlite-toggle-${WIDGET_CONFIG.widgetId}`);
        const closeBtn = document.getElementById(`chatlite-close-${WIDGET_CONFIG.widgetId}`);
        const chatWindow = document.getElementById(`chatlite-window-${WIDGET_CONFIG.widgetId}`);
        const sendBtn = document.getElementById(`chatlite-send-${WIDGET_CONFIG.widgetId}`);
        const input = document.getElementById(`chatlite-input-${WIDGET_CONFIG.widgetId}`);
        
        // Event listeners
        toggleBtn.addEventListener('click', toggleWidget);
        closeBtn.addEventListener('click', closeWidget);
        sendBtn.addEventListener('click', sendMessage);
        input.addEventListener('keypress', handleKeyPress);
        
        // Auto-open delay
        if (WIDGET_CONFIG.autoOpenDelay && WIDGET_CONFIG.autoOpenDelay > 0) {
            setTimeout(() => {
                if (!isOpen) openWidget();
            }, WIDGET_CONFIG.autoOpenDelay * 1000);
        }
        
        // Show widget
        widget.style.display = 'block';
        isInitialized = true;
        
        // Analytics tracking
        trackEvent('widget_loaded');
    }
    
    function toggleWidget() {
        if (isOpen) {
            closeWidget();
        } else {
            openWidget();
        }
    }
    
    function openWidget() {
        const chatWindow = document.getElementById(`chatlite-window-${WIDGET_CONFIG.widgetId}`);
        chatWindow.style.display = 'flex';
        isOpen = true;
        trackEvent('widget_opened');
    }
    
    function closeWidget() {
        const chatWindow = document.getElementById(`chatlite-window-${WIDGET_CONFIG.widgetId}`);
        chatWindow.style.display = 'none';
        isOpen = false;
        trackEvent('widget_closed');
    }
    
    function sendMessage() {
        const input = document.getElementById(`chatlite-input-${WIDGET_CONFIG.widgetId}`);
        const message = input.value.trim();
        
        if (!message) return;
        
        addMessage(message, 'user');
        input.value = '';
        
        // Send to backend
        sendToBackend(message);
        trackEvent('message_sent');
    }
    
    function handleKeyPress(event) {
        if (event.key === 'Enter') {
            sendMessage();
        }
    }
    
    function addMessage(content, type) {
        const messagesContainer = document.getElementById(`chatlite-messages-${WIDGET_CONFIG.widgetId}`);
        const messageDiv = document.createElement('div');
        messageDiv.className = `chatlite-message chatlite-${type}-message`;
        messageDiv.innerHTML = `<div class="chatlite-message-content">${escapeHtml(content)}</div>`;
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        if (WIDGET_CONFIG.enableSound && type === 'bot') {
            playNotificationSound();
        }
    }
    
    async function sendToBackend(message) {
        try {
            const response = await fetch(`${WIDGET_CONFIG.apiBaseUrl}/widget/chat/${WIDGET_CONFIG.widgetId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Origin': window.location.origin
                },
                body: JSON.stringify({
                    message: message,
                    visitor_id: getVisitorId(),
                    session_id: getSessionId(),
                    page_url: window.location.href,
                    page_title: document.title
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                addMessage(data.response, 'bot');
            } else {
                addMessage(WIDGET_CONFIG.offlineMessage || 'Sorry, we are currently offline.', 'bot');
            }
        } catch (error) {
            console.error('ChatLite error:', error);
            addMessage(WIDGET_CONFIG.offlineMessage || 'Sorry, we are currently offline.', 'bot');
        }
    }
    
    function getVisitorId() {
        let visitorId = localStorage.getItem('chatlite_visitor_id');
        if (!visitorId) {
            visitorId = 'visitor_' + Math.random().toString(36).substr(2, 9) + Date.now().toString(36);
            localStorage.setItem('chatlite_visitor_id', visitorId);
        }
        return visitorId;
    }
    
    function getSessionId() {
        let sessionId = sessionStorage.getItem('chatlite_session_id');
        if (!sessionId) {
            sessionId = 'session_' + Math.random().toString(36).substr(2, 9) + Date.now().toString(36);
            sessionStorage.setItem('chatlite_session_id', sessionId);
        }
        return sessionId;
    }
    
    function trackEvent(eventType) {
        try {
            fetch(`${WIDGET_CONFIG.apiBaseUrl}/widget/analytics/${WIDGET_CONFIG.widgetId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    event_type: eventType,
                    visitor_id: getVisitorId(),
                    session_id: getSessionId(),
                    page_url: window.location.href,
                    timestamp: new Date().toISOString()
                })
            }).catch(() => {}); // Fail silently
        } catch (error) {
            // Fail silently
        }
    }
    
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    function playNotificationSound() {
        try {
            const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkkBjiS2fLNeSsFJHfH8N2QQAoUXrPp66hVFApGn+DyvGkw==');
            audio.volume = 0.3;
            audio.play().catch(() => {}); // Fail silently
        } catch (error) {
            // Fail silently
        }
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initWidget);
    } else {
        initWidget();
    }
})();
        """.strip()
    
    async def generate_script(
        self, 
        website_id: str, 
        generated_by: Optional[str] = None
    ) -> Tuple[str, ScriptVersion]:
        """Generate a new script version for a website."""
        
        # Get website with current configuration
        result = await self.session.execute(
            select(Website).where(Website.id == website_id)
        )
        website = result.scalar_one_or_none()
        
        if not website:
            raise ValueError(f"Website not found: {website_id}")
        
        # Build configuration for script injection
        config = await self._build_script_config(website)
        
        # Generate script content
        script_content = await self._inject_config_into_template(config)
        
        # Calculate version number
        version_number = await self._get_next_version_number(website_id)
        
        # Generate content hash
        content_hash = self._generate_content_hash(script_content)
        
        # Generate script URL
        script_url = f"{self.cdn_base_url}/scripts/{website.widget_id}/v{version_number}.js"
        
        # Create script version record
        script_version = ScriptVersion(
            website_id=website_id,
            version_number=version_number,
            version_hash=content_hash,
            script_content=script_content,
            script_url=script_url,
            script_size=len(script_content.encode('utf-8')),
            widget_config=config,
            is_active=False,  # Not active until explicitly activated
            is_published=True,
            generated_by=generated_by,
            generation_time=0.0,  # Will be updated after generation
            cdn_uploaded=False,
            cache_key=f"script_{website.widget_id}_v{version_number}"
        )
        
        self.session.add(script_version)
        await self.session.commit()
        await self.session.refresh(script_version)
        
        # Update website with latest script info
        await self.session.execute(
            update(Website)
            .where(Website.id == website_id)
            .values(
                installation_script=script_content,
                script_version=f"v{version_number}",
                script_url=script_url,
                installation_status="generated",
                script_generated_at=datetime.utcnow()
            )
        )
        await self.session.commit()
        
        return script_content, script_version
    
    async def activate_script_version(self, website_id: str, version_id: str) -> bool:
        """Activate a specific script version."""
        
        # Deactivate all current versions
        await self.session.execute(
            update(ScriptVersion)
            .where(ScriptVersion.website_id == website_id)
            .values(is_active=False)
        )
        
        # Activate the specified version
        result = await self.session.execute(
            update(ScriptVersion)
            .where(
                ScriptVersion.id == version_id,
                ScriptVersion.website_id == website_id
            )
            .values(is_active=True)
        )
        
        if result.rowcount > 0:
            # Update website with active version info
            active_version = await self.session.execute(
                select(ScriptVersion).where(ScriptVersion.id == version_id)
            )
            version = active_version.scalar_one()
            
            await self.session.execute(
                update(Website)
                .where(Website.id == website_id)
                .values(
                    installation_script=version.script_content,
                    script_version=f"v{version.version_number}",
                    script_url=version.script_url,
                    installation_status="generated"
                )
            )
            
            await self.session.commit()
            return True
        
        return False
    
    async def get_script_versions(self, website_id: str) -> list[ScriptVersion]:
        """Get all script versions for a website."""
        result = await self.session.execute(
            select(ScriptVersion)
            .where(ScriptVersion.website_id == website_id)
            .order_by(ScriptVersion.version_number.desc())
        )
        return result.scalars().all()
    
    async def get_active_script(self, website_id: str) -> Optional[ScriptVersion]:
        """Get the currently active script version."""
        result = await self.session.execute(
            select(ScriptVersion)
            .where(
                ScriptVersion.website_id == website_id,
                ScriptVersion.is_active == True
            )
        )
        return result.scalar_one_or_none()
    
    async def _build_script_config(self, website: Website) -> Dict[str, Any]:
        """Build configuration object for script injection."""
        
        # Get allowed domains from URL
        allowed_domains = []
        if website.url:
            from urllib.parse import urlparse
            parsed = urlparse(website.url)
            if parsed.netloc:
                allowed_domains.append(parsed.netloc)
                # Also add without www if present
                if parsed.netloc.startswith('www.'):
                    allowed_domains.append(parsed.netloc[4:])
                else:
                    allowed_domains.append(f"www.{parsed.netloc}")
        
        config = {
            # Core identification
            "widgetId": website.widget_id,
            "websiteId": str(website.id),
            "allowedDomains": allowed_domains,
            
            # API configuration
            "apiBaseUrl": os.getenv("API_BASE_URL", "https://api.chatlite.app"),
            
            # Appearance settings
            "widgetColor": website.widget_color or "#0066CC",
            "widgetPosition": website.widget_position or "bottom-right",
            "widgetSize": website.widget_size or "medium",
            "widgetTheme": website.widget_theme or "light",
            "borderRadius": website.border_radius or 8,
            "fontFamily": website.font_family,
            
            # Behavior settings
            "showAvatar": website.show_avatar,
            "enableSound": website.enable_sound,
            "autoOpenDelay": website.auto_open_delay,
            "showOnlineStatus": website.show_online_status,
            
            # Messages
            "welcomeMessage": website.welcome_message,
            "placeholderText": website.placeholder_text or "Type your message...",
            "offlineMessage": website.offline_message or "We're currently offline",
            "thanksMessage": website.thanks_message or "Thanks for chatting!",
            
            # Branding
            "showBranding": website.show_branding,
            "customLogoUrl": website.custom_logo_url,
            "companyName": website.company_name or website.business_name or website.name,
            
            # Custom styling
            "customCSS": website.custom_css
        }
        
        # Remove None values
        return {k: v for k, v in config.items() if v is not None}
    
    async def _inject_config_into_template(self, config: Dict[str, Any]) -> str:
        """Inject configuration into JavaScript template."""
        config_json = json.dumps(config, indent=4)
        return self.script_template.replace(
            "/*CONFIG_PLACEHOLDER*/", 
            config_json
        )
    
    async def _get_next_version_number(self, website_id: str) -> int:
        """Get the next version number for a website."""
        result = await self.session.execute(
            select(ScriptVersion.version_number)
            .where(ScriptVersion.website_id == website_id)
            .order_by(ScriptVersion.version_number.desc())
            .limit(1)
        )
        
        last_version = result.scalar_one_or_none()
        return (last_version or 0) + 1
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate SHA-256 hash of script content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def prepare_for_cdn(self, script_version: ScriptVersion) -> Dict[str, Any]:
        """Prepare script version for CDN upload."""
        
        # Add cache headers and metadata
        cdn_metadata = {
            "content_type": "application/javascript",
            "cache_control": "public, max-age=86400",  # 24 hours
            "etag": script_version.version_hash,
            "content_encoding": "gzip",
            "expires": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
            "script_size": script_version.script_size,
            "version": script_version.version_number
        }
        
        # Update CDN metadata
        script_version.expires_at = datetime.utcnow() + timedelta(hours=24)
        
        await self.session.commit()
        
        return cdn_metadata
    
    async def mark_cdn_uploaded(self, script_version_id: str) -> bool:
        """Mark script version as successfully uploaded to CDN."""
        result = await self.session.execute(
            update(ScriptVersion)
            .where(ScriptVersion.id == script_version_id)
            .values(cdn_uploaded=True)
        )
        
        await self.session.commit()
        return result.rowcount > 0
    
    async def get_installation_code(self, website_id: str) -> str:
        """Get HTML installation code for a website."""
        
        active_script = await self.get_active_script(website_id)
        if not active_script:
            raise ValueError("No active script version found")
        
        return active_script.get_installation_code()
    
    async def detect_script_installation(self, website_id: str) -> Dict[str, Any]:
        """Detect if script is properly installed on website."""
        
        result = await self.session.execute(
            select(Website).where(Website.id == website_id)
        )
        website = result.scalar_one_or_none()
        
        if not website or not website.url:
            return {"installed": False, "error": "Website URL not available"}
        
        try:
            # This would typically involve crawling the website to check for script presence
            # For now, we'll return a placeholder implementation
            
            installation_status = {
                "installed": False,
                "script_found": False,
                "widget_id_match": False,
                "domain_authorized": True,
                "last_checked": datetime.utcnow().isoformat(),
                "errors": []
            }
            
            # Update website status
            await self.session.execute(
                update(Website)
                .where(Website.id == website_id)
                .values(script_last_checked=datetime.utcnow())
            )
            await self.session.commit()
            
            return installation_status
            
        except Exception as e:
            return {
                "installed": False, 
                "error": f"Installation check failed: {str(e)}"
            }