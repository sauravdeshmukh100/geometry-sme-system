#!/usr/bin/env python
"""
FastAPI Server for Geometry SME
Exposes all agent and tool capabilities via REST API
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
import logging
import os
from datetime import datetime

# Import our components
from src.agents.conversation_agent import ConversationAgent
from src.agents.task_router import TaskRouter
from src.llm.rag_llm_pipeline import GeometryTutorPipeline
from src.tools.tool_orchestrator import ToolOrchestrator
from src.tools.document_generator import DocumentGenerator
from src.tools.email_sender import EmailSender

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Geometry SME API",
    description="AI-powered Geometry Subject Matter Expert for K-12 Education",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (singleton pattern)
conversation_agent = None
tutor_pipeline = None
orchestrator = None
doc_generator = None
email_sender = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global conversation_agent, tutor_pipeline, orchestrator, doc_generator, email_sender
    
    logger.info("Initializing Geometry SME system...")
    
    try:
        # Initialize core components
        tutor_pipeline = GeometryTutorPipeline()
        orchestrator = ToolOrchestrator(tutor_pipeline=tutor_pipeline)
        conversation_agent = ConversationAgent(
            tutor_pipeline=tutor_pipeline,
            tool_orchestrator=orchestrator
        )
        doc_generator = DocumentGenerator()
        email_sender = EmailSender()
        
        logger.info("✓ Geometry SME system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}", exc_info=True)
        raise


# ========== Pydantic Models ==========

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=1000, description="User's message")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    include_context: bool = Field(True, description="Whether to retrieve RAG context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is the Pythagorean theorem?",
                "user_id": "user123",
                "include_context": True
            }
        }


class QuizRequest(BaseModel):
    """Request model for quiz generation."""
    topic: str = Field(..., min_length=1, max_length=100, description="Quiz topic")
    grade_level: str = Field(..., description="Target grade level")
    num_questions: int = Field(5, ge=1, le=20, description="Number of questions")
    format: Optional[str] = Field("pdf", description="Document format (pdf/docx/pptx)")
    
    @validator('grade_level')
    def validate_grade(cls, v):
        valid_grades = ['Grade 6', 'Grade 7', 'Grade 8', 'Grade 9', 'Grade 10']
        if v not in valid_grades:
            raise ValueError(f"Grade must be one of {valid_grades}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Triangles",
                "grade_level": "Grade 8",
                "num_questions": 5,
                "format": "pdf"
            }
        }


class QuizEmailRequest(QuizRequest):
    """Request model for quiz generation + email."""
    to_email: EmailStr = Field(..., description="Recipient email address")
    student_name: Optional[str] = Field(None, description="Student name")
    include_html: bool = Field(True, description="Use HTML email template")
    
    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Circles",
                "grade_level": "Grade 9",
                "num_questions": 10,
                "format": "pdf",
                "to_email": "student@school.edu",
                "student_name": "Alice"
            }
        }


class ExplanationRequest(BaseModel):
    """Request model for concept explanation."""
    concept: str = Field(..., min_length=1, max_length=200, description="Concept to explain")
    grade_level: str = Field(..., description="Target grade level")
    explanation_type: str = Field("step-by-step", description="Type of explanation")
    include_examples: bool = Field(True, description="Include worked examples")
    
    @validator('explanation_type')
    def validate_type(cls, v):
        valid_types = ['step-by-step', 'visual', 'proof', 'example']
        if v not in valid_types:
            raise ValueError(f"Type must be one of {valid_types}")
        return v
    
    class Config:
        json_schema_extra= {
            "example": {
                "concept": "Pythagorean Theorem",
                "grade_level": "Grade 9",
                "explanation_type": "step-by-step",
                "include_examples": True
            }
        }


class DocumentRequest(BaseModel):
    """Request model for document generation."""
    content_type: str = Field(..., description="Type of content (quiz/report/explanation)")
    format: str = Field(..., description="Document format (pdf/docx/pptx)")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content_type": "report",
                "format": "pdf",
                "title": "Geometry Study Guide",
                "content": "Chapter 1: Triangles...",
                "metadata": {"grade_level": "Grade 8"}
            }
        }


class EmailRequest(BaseModel):
    """Request model for sending email."""
    to_email: EmailStr = Field(..., description="Recipient email")
    subject: str = Field(..., min_length=1, max_length=200, description="Email subject")
    body: str = Field(..., min_length=1, description="Email body")
    attachments: Optional[List[str]] = Field(None, description="File paths to attach")
    html: bool = Field(False, description="Whether body is HTML")
    
    class Config:
        json_schema_extra = {
            "example": {
                "to_email": "student@school.edu",
                "subject": "Your Geometry Quiz",
                "body": "Here is your quiz on triangles...",
                "attachments": ["generated_docs/quiz_triangles.pdf"],
                "html": False
            }
        }


# ========== API Endpoints ==========

@app.get("/", tags=["General"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": "Geometry SME API",
        "version": "1.0.0",
        "description": "AI-powered Geometry tutor for K-12 education",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "chat": "/chat",
            "quiz": "/quiz/generate",
            "explanation": "/explain",
            "capabilities": "/capabilities"
        }
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    try:
        # Check if components are initialized
        if not all([conversation_agent, tutor_pipeline, orchestrator]):
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "unhealthy", "message": "System not fully initialized"}
            )
        
        # Get system statistics
        stats = tutor_pipeline.get_statistics()
        tool_status = orchestrator.get_tool_status()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "rag_status": stats.get('status'),
                "total_chunks": stats.get('rag_pipeline', {}).get('total_chunks', 0),
                "tools_available": tool_status.get('available_tools', {}),
                "document_formats": tool_status.get('document_formats', {})
            }
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/capabilities", tags=["General"])
async def get_capabilities():
    """Get system capabilities and features."""
    return conversation_agent.get_capabilities()


@app.post("/chat", tags=["Conversation"])
async def chat(request: ChatRequest):
    """
    Chat with the geometry tutor.
    Handles multi-turn conversations with memory.
    """
    try:
        response = conversation_agent.process_message(
            message=request.message,
            user_id=request.user_id
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}"
        )


@app.post("/chat/reset", tags=["Conversation"])
async def reset_conversation():
    """Reset conversation memory."""
    try:
        conversation_agent.reset_conversation()
        return {"message": "Conversation reset successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/quiz/generate", tags=["Quiz"])
async def generate_quiz(request: QuizRequest):
    """Generate a geometry quiz."""
    # print quiz topic and num_questions
    print(f"Generating quiz on topic: {request.topic} with {request.num_questions} questions")
    try:
        result = tutor_pipeline.generate_quiz(
            topic=request.topic,
            grade_level=request.grade_level,
            num_questions=request.num_questions
        )
        
        if not result.get('success'):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get('error', 'Quiz generation failed')
            )
        
        # Generate document if format specified
        if request.format:
            doc_path = doc_generator.generate(
                content_type='quiz',
                format=request.format,
                data=result
            )
            result['document_path'] = doc_path
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quiz generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/quiz/email", tags=["Quiz"])
async def generate_and_email_quiz(request: QuizEmailRequest, background_tasks: BackgroundTasks):
    """
    Generate a quiz and email it to a student.
    Uses background tasks for async processing.
    """
    try:
        # Execute workflow
        workflow_result = orchestrator.generate_and_email_quiz(
            topic=request.topic,
            grade_level=request.grade_level,
            to_email=request.to_email,
            num_questions=request.num_questions,
            format=request.format,
            student_name=request.student_name,
            include_html=request.include_html
        )
        
        return workflow_result.to_dict()
    
    except Exception as e:
        logger.error(f"Quiz email workflow error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/explain", tags=["Explanation"])
async def explain_concept(request: ExplanationRequest):
    """Generate a detailed explanation of a geometry concept."""
    try:
        result = tutor_pipeline.explain_concept(
            concept=request.concept,
            grade_level=request.grade_level,
            explanation_type=request.explanation_type,
            include_examples=request.include_examples
        )
        
        if not result.get('success'):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get('error', 'Explanation generation failed')
            )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/document/generate", tags=["Documents"])
async def generate_document(request: DocumentRequest):
    """Generate a document (PDF/DOCX/PPT)."""
    try:
        # Prepare data
        data = {
            'title': request.title,
            'content': request.content,
            'metadata': request.metadata or {}
        }
        
        # Generate document
        doc_path = doc_generator.generate(
            content_type=request.content_type,
            format=request.format,
            data=data
        )
        
        return {
            "success": True,
            "document_path": doc_path,
            "format": request.format,
            "download_url": f"/document/download?path={doc_path}"
        }
    
    except Exception as e:
        logger.error(f"Document generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/document/download", tags=["Documents"])
async def download_document(path: str):
    """Download a generated document."""
    try:
        if not os.path.exists(path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return FileResponse(
            path=path,
            filename=os.path.basename(path),
            media_type='application/octet-stream'
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/email/send", tags=["Email"])
async def send_email(request: EmailRequest):
    """Send an email with optional attachments."""
    try:
        result = email_sender.send_email(
            to_email=request.to_email,
            subject=request.subject,
            body=request.body,
            attachments=request.attachments,
            html=request.html
        )
        
        if not result.get('success'):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get('error', 'Email sending failed')
            )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Email error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/workflows", tags=["Workflows"])
async def list_workflows():
    """List available workflow templates."""
    return orchestrator.get_workflow_templates()


@app.post("/workflow/execute", tags=["Workflows"])
async def execute_workflow(workflow_name: str, parameters: Dict[str, Any]):
    """Execute a predefined workflow."""
    try:
        # This would route to appropriate workflow based on name
        # For now, return the workflow info
        templates = orchestrator.get_workflow_templates()
        
        workflow = next(
            (w for w in templates if w['name'] == workflow_name),
            None
        )
        
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_name}' not found"
            )
        
        return {
            "message": f"Workflow '{workflow_name}' would be executed",
            "parameters": parameters,
            "workflow": workflow
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ========== Error Handlers ==========

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url)
        }
    )


# ========== Run Server ==========

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )