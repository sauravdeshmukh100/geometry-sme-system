# src/llm/gemini_client.py

import os
import logging
from typing import List, Dict, Any, Optional
import time

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("Please install: pip install google-generativeai")

from ..config.settings import settings

logger = logging.getLogger(__name__)

class GeminiClient:
    """
    Client for Google Gemini API integration.
    Handles Q&A, content generation, and multi-step reasoning.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key. If None, reads from environment.
        """
        # Get API key from parameter, environment, or settings
        self.api_key = api_key or os.getenv('GEMINI_API_KEY') or getattr(settings, 'GEMINI_API_KEY', None)
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY in .env or pass as parameter.\n"
                "Get your key from: https://aistudio.google.com/app/api-keys"
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model_name = "gemini-2.5-flash"  # Fast and free
        try:
            self.model = genai.GenerativeModel("gemini-2.5-flash")
        except Exception as e:
            if "not found" in str(e).lower():
                self.model = genai.GenerativeModel("models/gemini-1.5-flash")
            else:
                raise
        
        # Generation config
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # Safety settings (allow educational content)
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        logger.info(f"Gemini client initialized with model: {self.model_name}")
        print(f"Gemini client initialized with model: {self.model_name}")
    
    def generate_answer(
        self,
        query: str,
        context: str,
        grade_level: Optional[str] = None,
        difficulty: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate an answer to a geometry query using retrieved context.
        
        Args:
            query: User's question
            context: Retrieved context from RAG
            grade_level: Target grade level (e.g., "Grade 7")
            difficulty: Content difficulty (Beginner/Intermediate/Advanced)
            max_retries: Number of retry attempts
        
        Returns:
            Dict with 'answer', 'sources', 'confidence', etc.
        """
        
        # Build prompt
        system_prompt = self._build_system_prompt(grade_level, difficulty)
        user_prompt = f"""Context from textbooks:
{context}

Student Question: {query}

Please provide a clear, step-by-step explanation appropriate for the student's level.
Include relevant formulas, diagrams descriptions, and examples where helpful."""

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Generate with retry logic
        for attempt in range(max_retries):
            try:
                start_time = time.time()

                response = self.model.generate_content(
                    full_prompt,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )

                elapsed_time = time.time() - start_time

                # Extract answer robustly
                answer = self._extract_response_text(response)

                # Check if response was blocked
                if not answer or "blocked" in answer.lower():
                    logger.warning(f"Response blocked or empty on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        answer = "I apologize, but I couldn't generate a response. Please try rephrasing your question."

                return {
                    'answer': answer,
                    'model': self.model_name,
                    'query': query,
                    'grade_level': grade_level,
                }
            except Exception as e:
                logger.exception("Error generating response on attempt %d: %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise
    
    def generate_quiz(
        self,
        topic: str,
        context: str,
        grade_level: str,
        num_questions: int = 5,
        question_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a quiz on a geometry topic.
        
        Args:
            topic: Topic for quiz (e.g., "Triangles", "Circles")
            context: Retrieved context about the topic
            grade_level: Target grade level
            num_questions: Number of questions to generate
            question_types: Types of questions (MCQ, Short Answer, etc.)
        
        Returns:
            Dict with quiz questions, answers, and metadata
        """
        
        if question_types is None:
            question_types = ["Multiple Choice", "Short Answer"]
        
        prompt = f"""You are a geometry teacher creating a quiz for {grade_level} students.

Topic: {topic}

Reference Material:
{context}

Create a quiz with {num_questions} questions covering key concepts from this topic.

Requirements:
1. Mix of question types: {', '.join(question_types)}
2. Progressive difficulty (easy to moderate)
3. Clear, unambiguous questions
4. Include diagrams descriptions where relevant
5. Provide correct answers with brief explanations

Format your response as:

Question 1: [Question text]
A) Option 1
B) Option 2
C) Option 3
D) Option 4
Correct Answer: [Letter]
Explanation: [Brief explanation]

Question 2: [For short answer questions, just ask the question]
Correct Answer: [Expected answer]
Explanation: [Brief explanation]

[Continue for all questions...]"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            quiz_content = self._extract_response_text(response)
            if quiz_content is None:
                raise RuntimeError("No text returned from Gemini response")
            
            return {
                'quiz': quiz_content,
                'topic': topic,
                'grade_level': grade_level,
                'num_questions': num_questions,
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error generating quiz: {e}")
            return {
                'quiz': None,
                'success': False,
                'error': str(e)
            }
    
    def generate_explanation(
        self,
        concept: str,
        context: str,
        grade_level: str,
        explanation_type: str = "step-by-step"
    ) -> Dict[str, Any]:
        """
        Generate a detailed explanation of a geometry concept.
        
        Args:
            concept: Concept to explain (e.g., "Pythagorean Theorem")
            context: Retrieved context
            grade_level: Target grade level
            explanation_type: "step-by-step", "visual", "proof", "example"
        
        Returns:
            Dict with explanation and metadata
        """
        
        explanation_prompts = {
            "step-by-step": "Provide a clear step-by-step explanation with examples.",
            "visual": "Describe visual representations and diagrams that help understand this concept.",
            "proof": "Provide a mathematical proof with detailed reasoning.",
            "example": "Explain through multiple worked examples with increasing complexity."
        }
        
        style = explanation_prompts.get(explanation_type, explanation_prompts["step-by-step"])
        
        prompt = f"""You are explaining geometry to {grade_level} students.

Concept: {concept}

Reference Material:
{context}

Task: {style}

Make it:
- Age-appropriate for {grade_level}
- Clear and easy to follow
- Include real-world applications if relevant
- Break down complex ideas into simple steps"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            explanation_text = self._extract_response_text(response)
            return {
                'explanation': explanation_text,
                'concept': concept,
                'grade_level': grade_level,
                'type': explanation_type,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return {
                'explanation': None,
                'success': False,
                'error': str(e)
            }
    
    def chat(
        self,
        message: str,
        chat_history: List[Dict[str, str]] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle conversational interaction with memory.
        
        Args:
            message: User's message
            chat_history: Previous conversation history
            context: Optional context from RAG
        
        Returns:
            Dict with response and updated history
        """
        
        if chat_history is None:
            chat_history = []
        
        # Build conversation context
        conversation = "Previous conversation:\n"
        for turn in chat_history[-5:]:  # Last 5 turns
            conversation += f"Student: {turn.get('user', '')}\n"
            conversation += f"Tutor: {turn.get('assistant', '')}\n\n"
        
        # Build prompt
        # avoid backslashes inside f-string expressions by preparing the context block separately
        context_block = f"Relevant information from textbooks:\n{context}\n" if context else ""

        prompt = (
            "You are a friendly geometry tutor helping a student.\n\n"
            f"{conversation}\n"
            f"{context_block}\n"
            f"Student: {message}\n\n"
            "Tutor:"
        )

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            assistant_message = self._extract_response_text(response) or "Sorry, I couldn't produce a reply."
            
            # Update history
            chat_history.append({
                'user': message,
                'assistant': assistant_message,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return {
                'response': assistant_message,
                'chat_history': chat_history,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {
                'response': "I'm sorry, I encountered an error. Could you rephrase that?",
                'success': False,
                'error': str(e)
            }
    
    def _build_system_prompt(
        self,
        grade_level: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> str:
        """Build system prompt based on context."""
        
        base_prompt = """You are an expert geometry tutor specializing in K-12 education (Grades 6-10).

Your role:
- Provide clear, accurate explanations of geometry concepts
- Use age-appropriate language and examples
- Break down complex problems into simple steps
- Include relevant formulas and theorems
- Encourage understanding, not just memorization

Guidelines:
- Always explain WHY something works, not just HOW
- Use real-world examples when possible
- If you're not sure, say so
- Encourage students to ask follow-up questions"""

        if grade_level:
            base_prompt += f"\n- Target audience: {grade_level} students"
        
        if difficulty:
            base_prompt += f"\n- Difficulty level: {difficulty}"
        
        return base_prompt

    def _extract_response_text(self, response) -> Optional[str]:
        """
        Robustly extract text from a Gemini response object.
        Tries common accessors and falls back to inspecting candidates/output.
        """
        # Log known finish reason if present
        try:
            finish_reason = getattr(response, "finish_reason", None)
            if not finish_reason and getattr(response, "candidates", None):
                first_cand = response.candidates[0]
                finish_reason = getattr(first_cand, "finish_reason", None) or getattr(first_cand, "metadata", {}).get("finish_reason")
            if finish_reason is not None:
                logger.debug(f"Gemini response finish_reason: {finish_reason}")
        except Exception:
            pass

        # 1) quick .text (may raise if no Part) - safe check
        try:
            text = getattr(response, "text", None)
            if text:
                return text
        except Exception:
            # can't use quick accessor
            pass

        # 2) candidates -> content -> text
        try:
            if getattr(response, "candidates", None):
                parts = []
                for cand in response.candidates:
                    # candidate.text sometimes exists
                    if getattr(cand, "text", None):
                        parts.append(cand.text)
                        continue
                    # candidate.content is list of parts
                    content = getattr(cand, "content", None) or getattr(cand, "output", None)
                    if content:
                        for p in content:
                            t = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None)
                            if t:
                                parts.append(t)
                if parts:
                    return "\n".join(parts)
        except Exception:
            pass

        # 3) output -> content -> text
        try:
            if getattr(response, "output", None):
                parts = []
                for out in response.output:
                    content = getattr(out, "content", None) or (out.get("content") if isinstance(out, dict) else None)
                    if content:
                        for p in content:
                            t = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None)
                            if t:
                                parts.append(t)
                if parts:
                    return "\n".join(parts)
        except Exception:
            pass

        # 4) as last resort try to stringify response dict for debugging
        try:
            if hasattr(response, "to_dict"):
                d = response.to_dict()
                import json
                return json.dumps(d)
        except Exception:
            pass

        return None
    
    def test_connection(self) -> bool:
        """Test if API key is valid and model is accessible."""
        try:
            response = self.model.generate_content(
                "Say 'Hello, I am ready to help with geometry!' in one sentence.",
                generation_config={"max_output_tokens": 50}
            )
            text = self._extract_response_text(response)
            if text:
                logger.info("✓ Gemini API connection successful")
                logger.info(f"Test response: {text}")
                return True
            else:
                logger.error("✗ Gemini API connection failed: no text returned in response")
                return False
        except Exception as e:
            logger.error(f"✗ Gemini API connection failed: {e}")
            return False