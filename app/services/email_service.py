"""
Email service for sending verification and notification emails.
Handles email templates and sending functionality.
"""

import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import logging

from app.core.config import get_settings
from app.models.user import User

settings = get_settings()
logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending various types of emails."""

    def __init__(self):
        self.settings = settings
        self.smtp_server = getattr(settings, 'SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = getattr(settings, 'SMTP_PORT', 587)
        self.smtp_username = getattr(settings, 'SMTP_USERNAME', None)
        self.smtp_password = getattr(settings, 'SMTP_PASSWORD', None)
        self.from_email = getattr(settings, 'FROM_EMAIL', 'noreply@litechat.com')

    def _create_smtp_connection(self):
        """Create SMTP connection for sending emails."""
        if not self.smtp_username or not self.smtp_password:
            logger.warning("SMTP credentials not configured - emails will be logged instead of sent")
            return None
            
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            return server
        except Exception as e:
            logger.error(f"Failed to create SMTP connection: {e}")
            return None

    def _send_email(
        self, 
        to_email: str, 
        subject: str, 
        html_content: str, 
        text_content: Optional[str] = None
    ) -> bool:
        """
        Send email with HTML and optional text content.
        Returns True if successful, False otherwise.
        """
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = to_email

            # Add text version if provided
            if text_content:
                text_part = MIMEText(text_content, 'plain')
                msg.attach(text_part)

            # Add HTML version
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

            # Send email
            server = self._create_smtp_connection()
            if server:
                server.send_message(msg)
                server.quit()
                logger.info(f"Email sent successfully to {to_email}")
                return True
            else:
                # For development/testing - log email instead of sending
                logger.info(f"EMAIL (to {to_email}): {subject}")
                logger.info(f"HTML Content: {html_content}")
                return True

        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False

    def send_verification_email(self, user: User, verification_url: str) -> bool:
        """
        Send email verification email to new user.
        Returns True if email was sent successfully.
        """
        subject = "Welcome to LiteChat - Please verify your email"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #0066CC; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .button {{ background: #0066CC; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 20px 0; }}
                .footer {{ background: #f5f5f5; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Welcome to LiteChat!</h1>
                </div>
                <div class="content">
                    <h2>Hello {user.name}!</h2>
                    <p>Thank you for signing up for LiteChat. To complete your registration and start adding AI-powered chatbots to your websites, please verify your email address.</p>
                    
                    <p>Click the button below to verify your email:</p>
                    <a href="{verification_url}" class="button">Verify Email Address</a>
                    
                    <p>Or copy and paste this link into your browser:</p>
                    <p><a href="{verification_url}">{verification_url}</a></p>
                    
                    <p>Once verified, you'll be able to:</p>
                    <ul>
                        <li>Register your websites for chatbot integration</li>
                        <li>Customize your chatbot appearance and behavior</li>
                        <li>Generate installation scripts for easy deployment</li>
                        <li>Access analytics and conversation insights</li>
                    </ul>
                    
                    <p>If you didn't create this account, please ignore this email.</p>
                </div>
                <div class="footer">
                    <p>© 2025 LiteChat. All rights reserved.</p>
                    <p>This email was sent to {user.email}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        Welcome to LiteChat!
        
        Hello {user.name}!
        
        Thank you for signing up for LiteChat. To complete your registration and start adding AI-powered chatbots to your websites, please verify your email address.
        
        Please visit this link to verify your email:
        {verification_url}
        
        Once verified, you'll be able to register your websites, customize your chatbot, generate installation scripts, and access analytics.
        
        If you didn't create this account, please ignore this email.
        
        © 2025 LiteChat. All rights reserved.
        """
        
        return self._send_email(user.email, subject, html_content, text_content)

    def send_password_reset_email(self, user: User, reset_url: str) -> bool:
        """
        Send password reset email to user.
        Returns True if email was sent successfully.
        """
        subject = "LiteChat - Password Reset Request"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #0066CC; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .button {{ background: #0066CC; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 20px 0; }}
                .warning {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                .footer {{ background: #f5f5f5; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Password Reset Request</h1>
                </div>
                <div class="content">
                    <h2>Hello {user.name}!</h2>
                    <p>We received a request to reset the password for your LiteChat account.</p>
                    
                    <p>Click the button below to reset your password:</p>
                    <a href="{reset_url}" class="button">Reset Password</a>
                    
                    <p>Or copy and paste this link into your browser:</p>
                    <p><a href="{reset_url}">{reset_url}</a></p>
                    
                    <div class="warning">
                        <strong>Security Notice:</strong>
                        <ul>
                            <li>This link will expire in 1 hour for security</li>
                            <li>If you didn't request this reset, please ignore this email</li>
                            <li>Your account will remain secure and unchanged</li>
                        </ul>
                    </div>
                </div>
                <div class="footer">
                    <p>© 2025 LiteChat. All rights reserved.</p>
                    <p>This email was sent to {user.email}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        Password Reset Request
        
        Hello {user.name}!
        
        We received a request to reset the password for your LiteChat account.
        
        Please visit this link to reset your password:
        {reset_url}
        
        Security Notice:
        - This link will expire in 1 hour for security
        - If you didn't request this reset, please ignore this email
        - Your account will remain secure and unchanged
        
        © 2025 LiteChat. All rights reserved.
        """
        
        return self._send_email(user.email, subject, html_content, text_content)

    def send_welcome_email(self, user: User) -> bool:
        """
        Send welcome email after email verification.
        Returns True if email was sent successfully.
        """
        subject = "Welcome to LiteChat - Your account is ready!"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #28a745; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .button {{ background: #0066CC; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 20px 0; }}
                .steps {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                .footer {{ background: #f5f5f5; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎉 Account Verified!</h1>
                </div>
                <div class="content">
                    <h2>Congratulations {user.name}!</h2>
                    <p>Your LiteChat account has been successfully verified and is ready to use.</p>
                    
                    <div class="steps">
                        <h3>Next Steps:</h3>
                        <ol>
                            <li><strong>Register Your Website</strong> - Add your website URL and we'll analyze its content</li>
                            <li><strong>Customize Your Chatbot</strong> - Choose colors, position, and welcome messages</li>
                            <li><strong>Install the Script</strong> - Get a simple JavaScript snippet to add to your website</li>
                            <li><strong>Monitor Performance</strong> - Track conversations and chatbot effectiveness</li>
                        </ol>
                    </div>
                    
                    <p>Ready to get started? Login to your dashboard:</p>
                    <a href="{settings.frontend_url}/login" class="button">Go to Dashboard</a>
                    
                    <p>If you have any questions, feel free to reach out to our support team.</p>
                </div>
                <div class="footer">
                    <p>© 2025 LiteChat. All rights reserved.</p>
                    <p>This email was sent to {user.email}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        Account Verified!
        
        Congratulations {user.name}!
        
        Your LiteChat account has been successfully verified and is ready to use.
        
        Next Steps:
        1. Register Your Website - Add your website URL and we'll analyze its content
        2. Customize Your Chatbot - Choose colors, position, and welcome messages  
        3. Install the Script - Get a simple JavaScript snippet to add to your website
        4. Monitor Performance - Track conversations and chatbot effectiveness
        
        Ready to get started? Login to your dashboard:
        {settings.frontend_url}/login
        
        If you have any questions, feel free to reach out to our support team.
        
        © 2025 LiteChat. All rights reserved.
        """
        
        return self._send_email(user.email, subject, html_content, text_content)


# Global email service instance
email_service = EmailService()

# Convenience functions for sending emails
async def send_verification_email(user: User, verification_url: str) -> bool:
    """Send verification email asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, email_service.send_verification_email, user, verification_url)

async def send_password_reset_email(user: User, reset_url: str) -> bool:
    """Send password reset email asynchronously.""" 
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, email_service.send_password_reset_email, user, reset_url)

async def send_welcome_email(user: User) -> bool:
    """Send welcome email asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, email_service.send_welcome_email, user)