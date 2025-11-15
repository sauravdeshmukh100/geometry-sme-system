#!/usr/bin/env python
"""
Email Sender Tool
Handles sending generated documents and reports via email
"""

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.utils import formataddr
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from ..config.settings import settings

logger = logging.getLogger(__name__)

print(f"EMAIL_USERNAME: {settings.EMAIL_USERNAME}")
print(f"EMAIL_PASSWORD: {settings.EMAIL_PASSWORD}")
print(f"SMTP_HOST: {settings.SMTP_HOST}")


class EmailError(Exception):
    """Custom exception for email errors."""
    pass


class EmailSender:
    """
    Email sender for geometry tutoring documents.
    Supports attachments, HTML emails, and multiple recipients.
    """
    
    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True,
        from_name: str = "LLMApp"
    ):
        """
        Initialize email sender.
        
        Args:
            smtp_server: SMTP server address (e.g., smtp.gmail.com)
            smtp_port: SMTP port (default: 587 for TLS, 465 for SSL)
            username: Email username
            password: Email password or app password
            use_tls: Whether to use TLS (default True)  
            from_name: Display name for sender
        """
        # Get credentials from environment via settings (Pydantic BaseSettings)
        self.smtp_server = smtp_server or getattr(settings, 'SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = smtp_port or getattr(settings, 'SMTP_PORT', 587 if use_tls else 465)
        self.username = username or getattr(settings, 'EMAIL_USERNAME', None)
        self.password = password or getattr(settings, 'EMAIL_PASSWORD', None)
        self.use_tls = use_tls
        self.from_name = from_name
        
        if not self.username or not self.password:
            logger.warning("Email credentials not configured. Set EMAIL_USERNAME and EMAIL_PASSWORD in settings.")
        
        logger.info(f"Email Sender initialized: {self.smtp_server}:{self.smtp_port}")
    
    def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        attachments: Optional[List[str]] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        html: bool = False,
        reply_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send an email with optional attachments.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body (plain text or HTML)
            attachments: List of file paths to attach
            cc: CC recipients
            bcc: BCC recipients
            html: Whether body is HTML (default False)
            reply_to: Reply-to email address
        
        Returns:
            Dict with success status and message
        """
        if not self.username or not self.password:
            return {
                'success': False,
                'error': 'Email credentials not configured. Set EMAIL_USERNAME and EMAIL_PASSWORD in environment.'
            }
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = formataddr((self.from_name, self.username))
            msg['To'] = to_email
            msg['Subject'] = subject
            msg['Date'] = datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')
            
            if cc:
                msg['Cc'] = ', '.join(cc)
            
            if reply_to:
                msg['Reply-To'] = reply_to
            
            # Attach body
            body_type = 'html' if html else 'plain'
            msg.attach(MIMEText(body, body_type))
            
            # Attach files
            if attachments:
                for filepath in attachments:
                    if not os.path.exists(filepath):
                        logger.warning(f"Attachment not found: {filepath}")
                        continue
                    
                    try:
                        with open(filepath, 'rb') as f:
                            filename = Path(filepath).name
                            
                            # Determine MIME type
                            if filepath.endswith('.pdf'):
                                mime_type = 'application/pdf'
                            elif filepath.endswith('.docx'):
                                mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                            elif filepath.endswith('.pptx'):
                                mime_type = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
                            else:
                                mime_type = 'application/octet-stream'
                            
                            part = MIMEApplication(f.read(), mime_type)
                            part.add_header('Content-Disposition', 'attachment', filename=filename)
                            msg.attach(part)
                            
                            logger.info(f"Attached file: {filename}")
                    
                    except Exception as e:
                        logger.error(f"Error attaching file {filepath}: {e}")
            
            # Send email
            all_recipients = [to_email]
            if cc:
                all_recipients.extend(cc)
            if bcc:
                all_recipients.extend(bcc)
            
            with self._get_smtp_connection() as server:
                server.sendmail(self.username, all_recipients, msg.as_string())
            
            logger.info(f"✓ Email sent to {to_email}")
            
            return {
                'success': True,
                'message': f'Email sent successfully to {to_email}',
                'to': to_email,
                'subject': subject,
                'attachments': len(attachments) if attachments else 0,
                'sent_at': datetime.now().isoformat()
            }
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed: {e}")
            return {
                'success': False,
                'error': 'Authentication failed. Check email credentials.',
                'details': str(e)
            }
        
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return {
                'success': False,
                'error': f'Failed to send email: {str(e)}'
            }
        
        except Exception as e:
            logger.error(f"Error sending email: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}'
            }
    
    def send_quiz(
        self,
        to_email: str,
        quiz_data: Dict[str, Any],
        attachments: Optional[List[str]] = None,
        student_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a quiz to a student via email.
        
        Args:
            to_email: Student email address
            quiz_data: Quiz data (topic, grade_level, etc.)
            attachments: List of file paths (PDF/DOCX of quiz)
            student_name: Student name for personalization
        
        Returns:
            Result dict
        """
        topic = quiz_data.get('topic', 'Geometry')
        grade = quiz_data.get('grade_level', 'N/A')
        
        # Compose subject
        subject = f"Geometry Quiz: {topic} ({grade})"
        
        # Compose body
        greeting = f"Hi {student_name}," if student_name else "Hello,"
        
        body = f"""{greeting}

Here is your geometry quiz on {topic} for {grade}.

Quiz Details:
- Topic: {topic}
- Grade Level: {grade}
- Questions: {quiz_data.get('num_questions', 'N/A')}

Please complete the quiz and submit your answers. Good luck!

Best regards,
Geometry Tutor

---
This is an automated message from the Geometry SME system.
"""
        
        return self.send_email(
            to_email=to_email,
            subject=subject,
            body=body,
            attachments=attachments
        )
    
    def send_report(
        self,
        to_email: str,
        report_title: str,
        report_summary: str,
        attachments: Optional[List[str]] = None,
        recipient_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a report via email.
        
        Args:
            to_email: Recipient email
            report_title: Report title
            report_summary: Brief summary of report
            attachments: Report file paths
            recipient_name: Recipient name
        
        Returns:
            Result dict
        """
        subject = f"Report: {report_title}"
        
        greeting = f"Hi {recipient_name}," if recipient_name else "Hello,"
        
        body = f"""{greeting}

Please find attached the report: {report_title}

Summary:
{report_summary}

The complete report is attached to this email.

Best regards,
Geometry Tutor

---
Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
"""
        
        return self.send_email(
            to_email=to_email,
            subject=subject,
            body=body,
            attachments=attachments
        )
    
    def send_explanation(
        self,
        to_email: str,
        concept: str,
        explanation_preview: str,
        attachments: Optional[List[str]] = None,
        student_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a concept explanation via email.
        
        Args:
            to_email: Student email
            concept: Concept name
            explanation_preview: Short preview of explanation
            attachments: Document file paths
            student_name: Student name
        
        Returns:
            Result dict
        """
        subject = f"Geometry Explanation: {concept}"
        
        greeting = f"Hi {student_name}," if student_name else "Hello,"
        
        # Truncate preview
        if len(explanation_preview) > 300:
            explanation_preview = explanation_preview[:300] + "..."
        
        body = f"""{greeting}

Here is an explanation of the geometry concept: {concept}

Preview:
{explanation_preview}

Please see the attached document for the complete explanation with examples and diagrams.

Keep learning!

Best regards,
Geometry Tutor
"""
        
        return self.send_email(
            to_email=to_email,
            subject=subject,
            body=body,
            attachments=attachments
        )
    
    def send_batch_notification(
        self,
        to_emails: List[str],
        subject: str,
        body: str,
        attachments: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Send the same email to multiple recipients.
        
        Args:
            to_emails: List of recipient emails
            subject: Email subject
            body: Email body
            attachments: Optional attachments
        
        Returns:
            Dict with results for each recipient
        """
        results = {
            'total': len(to_emails),
            'success': 0,
            'failed': 0,
            'details': []
        }
        
        for email in to_emails:
            result = self.send_email(
                to_email=email,
                subject=subject,
                body=body,
                attachments=attachments
            )
            
            if result['success']:
                results['success'] += 1
            else:
                results['failed'] += 1
            
            results['details'].append({
                'email': email,
                'success': result['success'],
                'error': result.get('error')
            })
        
        logger.info(f"Batch email complete: {results['success']}/{results['total']} sent")
        
        return results
    
    def _get_smtp_connection(self):
        """Create SMTP connection with proper configuration."""
        if self.use_tls:
            # Use STARTTLS
            server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10)
            server.ehlo()
            server.starttls()
            server.ehlo()
        else:
            # Use SSL
            server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, timeout=10)
        
        server.login(self.username, self.password)
        return server
    
    def test_connection(self) -> bool:
        """
        Test SMTP connection and credentials.
        
        Returns:
            True if connection successful
        """
        if not self.username or not self.password:
            logger.error("Email credentials not configured")
            return False
        
        try:
            with self._get_smtp_connection() as server:
                logger.info("✓ Email connection test successful")
                return True
        
        except smtplib.SMTPAuthenticationError:
            logger.error("✗ Email authentication failed")
            return False
        
        except Exception as e:
            logger.error(f"✗ Email connection test failed: {e}")
            return False


# HTML Email Templates
def create_html_quiz_email(
    quiz_data: Dict[str, Any],
    student_name: Optional[str] = None
) -> str:
    """Create HTML email for quiz notification."""
    
    topic = quiz_data.get('topic', 'Geometry')
    grade = quiz_data.get('grade_level', 'N/A')
    num_questions = quiz_data.get('num_questions', 'N/A')
    
    greeting = f"Hi {student_name}," if student_name else "Hello,"
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #1a73e8; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
        .content {{ background-color: #f8f9fa; padding: 20px; border-radius: 0 0 8px 8px; }}
        .quiz-details {{ background-color: white; padding: 15px; border-left: 4px solid #1a73e8; margin: 20px 0; }}
        .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📐 Geometry Quiz</h1>
        </div>
        <div class="content">
            <p>{greeting}</p>
            
            <p>Here is your geometry quiz on <strong>{topic}</strong> for <strong>{grade}</strong>.</p>
            
            <div class="quiz-details">
                <h3>Quiz Details</h3>
                <ul>
                    <li><strong>Topic:</strong> {topic}</li>
                    <li><strong>Grade Level:</strong> {grade}</li>
                    <li><strong>Questions:</strong> {num_questions}</li>
                </ul>
            </div>
            
            <p>Please complete the quiz and submit your answers. The quiz is attached to this email.</p>
            
            <p><strong>Good luck! 🎓</strong></p>
            
            <p>Best regards,<br>Geometry Tutor</p>
        </div>
        <div class="footer">
            <p>This is an automated message from the Geometry SME system.<br>
            Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
    </div>
</body>
</html>
"""
    return html


def create_html_report_email(
    report_title: str,
    report_summary: str,
    recipient_name: Optional[str] = None
) -> str:
    """Create HTML email for report notification."""
    
    greeting = f"Hi {recipient_name}," if recipient_name else "Hello,"
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #34a853; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
        .content {{ background-color: #f8f9fa; padding: 20px; border-radius: 0 0 8px 8px; }}
        .summary {{ background-color: white; padding: 15px; border-left: 4px solid #34a853; margin: 20px 0; }}
        .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Report Ready</h1>
        </div>
        <div class="content">
            <p>{greeting}</p>
            
            <p>Your report is ready: <strong>{report_title}</strong></p>
            
            <div class="summary">
                <h3>Summary</h3>
                <p>{report_summary}</p>
            </div>
            
            <p>The complete report is attached to this email.</p>
            
            <p>Best regards,<br>Geometry Tutor</p>
        </div>
        <div class="footer">
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
    </div>
</body>
</html>
"""
    return html