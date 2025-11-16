#!/usr/bin/env python
"""
Merged Gemini Client - Best of Both Versions
Combines robust error handling with structured outputs
"""

from curses import raw
import os
import logging
import json
import time
from typing import List, Dict, Any, Optional

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("Please install: pip install google-generativeai")

from ..config.settings import settings

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Enhanced Gemini client with robust error handling and structured outputs.
    Supports Q&A, quiz generation, explanations, and chat conversations.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key. If None, reads from environment/settings.
        """
        # Get API key with fallback chain
        self.api_key = (
            api_key or 
            os.getenv('GEMINI_API_KEY') or 
            getattr(settings, 'GEMINI_API_KEY', None)
        )
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY in .env or pass as parameter.\n"
                "Get your key from: https://aistudio.google.com/app/apikey"
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize model with fallback
        self.model_name = getattr(settings, 'GEMINI_MODEL', 'gemini-2.0-flash-exp')
        self.model = self._initialize_model()
        
        # Generation config
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 20000,
        }
        
        # Safety settings (allow educational content)
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        # System instruction templates
        self.system_instructions = self._load_system_instructions()
        
        logger.info(f"✓ Gemini client initialized with model: {self.model_name}")
        print(f"✓ Gemini client initialized with model: {self.model_name}")
    
    def _initialize_model(self) -> genai.GenerativeModel:
        """Initialize model with fallback to alternative versions."""
        try:
            return genai.GenerativeModel(self.model_name)
        except Exception as e:
            logger.warning(f"Model {self.model_name} not available, trying fallbacks...")
            
            # Try fallback models
            fallback_models = [
                "gemini-2.0-flash-exp",
                "gemini-1.5-flash",
                "models/gemini-1.5-flash"
            ]
            
            for model_name in fallback_models:
                try:
                    logger.info(f"Trying model: {model_name}")
                    self.model_name = model_name
                    return genai.GenerativeModel(model_name)
                except Exception:
                    continue
            
            raise RuntimeError(f"Could not initialize any Gemini model: {e}")
    
    def _load_system_instructions(self) -> Dict[str, str]:
        """Load system instruction templates for different tasks."""
        return {
            "qa": """You are an expert geometry tutor for K-12 students (grades 6-10).

Your role:
- Provide clear, accurate explanations of geometry concepts
- Use age-appropriate language and examples
- Break down complex problems into simple steps
- Include relevant formulas and theorems
- Encourage understanding, not just memorization

Guidelines:
- Always explain WHY something works, not just HOW
- Use real-world examples when possible
- Base answers on the provided context from textbooks
- If you're not sure, say so honestly
- Encourage students to ask follow-up questions""",
            
            "quiz": """You are an expert geometry quiz creator for K-12 students.

Your role:
- Create grade-appropriate questions
- Cover various difficulty levels
- Include clear, unambiguous questions
- Provide accurate answers and explanations
- Focus on understanding, not just facts

Requirements:
- Test conceptual understanding
- Progressive difficulty
- Clear question format
- Brief but accurate explanations""",
            
            "explanation": """You are an expert geometry teacher creating detailed explanations.

Your role:
- Break down complex topics into simple steps
- Use analogies and real-world examples
- Provide visual descriptions where helpful
- Address common misconceptions
- Build on prerequisite knowledge

Make explanations engaging and easy to understand.""",
            
            "chat": """You are a friendly geometry tutor helping a student.

Your role:
- Maintain conversational tone
- Remember context from chat history
- Adapt to the student's level
- Be encouraging and supportive
- Guide students to understand, not just give answers"""
        }
    
    # ========== Core Generation Methods ==========
    
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
            Dict with 'answer', 'model', 'success', etc.
        """
        # Build system prompt
        system_prompt = self._build_system_prompt("qa", grade_level, difficulty)
        
        # Build user prompt
        user_prompt = f"""Context from textbooks:
{context}

Student Question: {query}

Please provide a clear, step-by-step explanation appropriate for the student's level.
Include relevant formulas, diagram descriptions, and examples where helpful."""

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
                
                # Check if response was blocked or empty
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
                    'generation_time': elapsed_time,
                    'success': True
                }
                
            except Exception as e:
                logger.exception(f"Error generating response on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    return {
                        'answer': f"An error occurred: {str(e)}",
                        'success': False,
                        'error': str(e)
                    }
    
    def generate_quiz(
        self,
        topic: str,
        context: str,
        grade_level: str,
        num_questions: int = 5,
        question_types: Optional[List[str]] = None,
        use_structured_output: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a quiz on a geometry topic.
        
        Args:
            topic: Topic for quiz (e.g., "Triangles", "Circles")
            context: Retrieved context about the topic
            grade_level: Target grade level
            num_questions: Number of questions to generate
            question_types: Types of questions (MCQ, Short Answer, etc.)
            use_structured_output: Return structured JSON format
        
        Returns:
            Dict with quiz questions, answers, and metadata
        """
        if question_types is None:
            question_types = ["Multiple Choice", "Short Answer"]
        
        system_instruction = self._build_system_prompt("quiz", grade_level)
        
        if use_structured_output:
            # Use structured JSON output
            return self._generate_structured_quiz(
                topic, context, grade_level, num_questions, 
                question_types, system_instruction
            )
        else:
            # Use freeform text output
            return self._generate_freeform_quiz(
                topic, context, grade_level, num_questions,
                question_types, system_instruction
            )
    
    def _generate_structured_quiz(
        self,
        topic: str,
        context: str,
        grade_level: str,
        num_questions: int,
        question_types: List[str],
        system_instruction: str
    ) -> Dict[str, Any]:
        """Generate quiz with structured JSON output."""
        
        prompt = f"""Based on the following geometry content about {topic}, create a quiz with {num_questions} questions.

Context:
{context}

Requirements:
- Grade level: {grade_level}
- Question types: {', '.join(question_types)}
- Progressive difficulty (easy to moderate)
- Include clear options for MCQs
- Provide correct answers with brief explanations"""
        
        # Define output schema
        schema = {
            "quiz_title": "string",
            "topic": "string",
            "grade_level": "string",
            "questions": [
                {
                    "question_number": "int",
                    "question_text": "string",
                    "question_type": "string (Multiple Choice / Short Answer)",
                    "options": ["A) text", "B) text", "C) text", "D) text"],
                    "correct_answer": "string",
                    "explanation": "string"
                }
            ]
        }
        
        try:
            quiz_data = self.generate_structured_output(
                prompt=prompt,
                system_instruction=system_instruction,
                output_schema=schema
            )
            
            quiz_data["metadata"] = {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": self.model_name
            }
            quiz_data["success"] = True
            
            return quiz_data
            
        except Exception as e:
            logger.error(f"Structured quiz generation failed: {e}")
            # Fallback to freeform
            logger.info("Falling back to freeform quiz generation")
            return self._generate_freeform_quiz(
                topic, context, grade_level, num_questions,
                question_types, system_instruction
            )
    
    def _generate_freeform_quiz(
        self,
        topic: str,
        context: str,
        grade_level: str,
        num_questions: int,
        question_types: List[str],
        system_instruction: str
    ) -> Dict[str, Any]:
        """Generate quiz with freeform text output."""
        
        prompt = f"""You are a geometry teacher creating a quiz for {grade_level} students.

Topic: {topic}

Reference Material:
{context}

Create a quiz with {num_questions} questions covering key concepts from this topic.

Requirements:
1. Mix of question types: {', '.join(question_types)}
2. Progressive difficulty (easy to moderate)
3. Clear, unambiguous questions
4. Include diagram descriptions where relevant
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
                f"{system_instruction}\n\n{prompt}",
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
                'model': self.model_name,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error generating freeform quiz: {e}")
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
        explanation_type: str = "step-by-step",
        include_examples: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a detailed explanation of a geometry concept.
        
        Args:
            concept: Concept to explain (e.g., "Pythagorean Theorem")
            context: Retrieved context
            grade_level: Target grade level
            explanation_type: "step-by-step", "visual", "proof", "example"
            include_examples: Whether to include worked examples
        
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
        
        system_instruction = self._build_system_prompt("explanation", grade_level)
        
        prompt = f"""Explain the geometry concept: {concept}

Context from textbooks:
{context}

Task: {style}

Requirements:
- Target grade: {grade_level}
- Start with a simple definition
- {"Include 2-3 worked examples" if include_examples else "Focus on conceptual understanding"}
- Use analogies and real-world applications where relevant
- Break down complex ideas into simple steps
- Highlight common mistakes to avoid"""

        try:
            response = self.model.generate_content(
                f"{system_instruction}\n\n{prompt}",
                generation_config={**self.generation_config, "temperature": 0.8},
                safety_settings=self.safety_settings
            )
            
            explanation_text = self._extract_response_text(response)
            
            return {
                'explanation': explanation_text,
                'concept': concept,
                'grade_level': grade_level,
                'type': explanation_type,
                'model': self.model_name,
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
        chat_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[str] = None,
        grade_level: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle conversational interaction with memory.
        
        Args:
            message: User's message
            chat_history: Previous conversation history
            context: Optional context from RAG
            grade_level: Optional grade level for adaptation
        
        Returns:
            Dict with response and updated history
        """
        if chat_history is None:
            chat_history = []
        
        # Build conversation context
        conversation = "Previous conversation:\n"
        for turn in chat_history[-5:]:  # Last 5 turns for context window
            conversation += f"Student: {turn.get('user', '')}\n"
            conversation += f"Tutor: {turn.get('assistant', '')}\n\n"
        
        # Build system instruction
        system_instruction = self._build_system_prompt("chat", grade_level)
        
        # Build context block if available
        context_block = f"\nRelevant information from textbooks:\n{context}\n" if context else ""
        
        # Build full prompt
        prompt = (
            f"{conversation}\n"
            f"{context_block}\n"
            f"Student: {message}\n\n"
            "Tutor:"
        )

        try:
            response = self.model.generate_content(
                f"{system_instruction}\n\n{prompt}",
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            assistant_message = self._extract_response_text(response)
            
            if not assistant_message:
                assistant_message = "Sorry, I couldn't produce a reply. Could you rephrase that?"
            
            # Update history
            chat_history.append({
                'user': message,
                'assistant': assistant_message,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return {
                'response': assistant_message,
                'chat_history': chat_history,
                'model': self.model_name,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {
                'response': "I'm sorry, I encountered an error. Could you rephrase that?",
                'chat_history': chat_history,
                'success': False,
                'error': str(e)
            }
    
    # ========== Utility Methods ==========
    
    def generate_structured_output(
        self,
        prompt: str,
        system_instruction: str,
        output_schema: Dict[str, Any],
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output.
        
        Args:
            prompt: User prompt
            system_instruction: System instruction
            output_schema: Expected JSON schema
            temperature: Lower for more structured output
        
        Returns:
            Parsed JSON response
        """
        # Add JSON format instruction
        full_prompt = f"""{prompt}

IMPORTANT: Respond ONLY with valid JSON matching this schema:
{json.dumps(output_schema, indent=2)}

Do not include any markdown formatting, backticks, or extra text. Just pure JSON."""
        
        try:
            response = self.model.generate_content(
                f"{system_instruction}\n\n{full_prompt}",
                generation_config={
                    **self.generation_config,
                    "temperature": temperature
                },
                safety_settings=self.safety_settings
            )
            
            print(f"Raw Gemini response: {response}")
            
            response_text = self._extract_response_text(response)
            print(f"Extracted response text for JSON parsing: {response_text}")
            
            if not response_text:
                raise ValueError("Empty response from model")
            
            # Clean response (remove markdown if present)
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            return json.loads(response_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Response text: {response_text}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Structured output generation failed: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            result = self.model.count_tokens(text)
            return result.total_tokens
        except Exception:
            # Fallback: rough estimate (1 token ≈ 0.75 words)
            return int(len(text.split()) * 1.3)
    
    def _build_system_prompt(
        self,
        task_type: str,
        grade_level: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> str:
        """Build system prompt based on task type and context."""
        
        base_prompt = self.system_instructions.get(task_type, self.system_instructions["qa"])
        
        if grade_level:
            base_prompt += f"\n\nTarget audience: {grade_level} students"
        
        if difficulty:
            base_prompt += f"\nDifficulty level: {difficulty}"
        
        return base_prompt
    
    def _extract_response_text(self, response) -> Optional[str]:
        """
        Robustly extract text from a Gemini response object.
        Tries multiple accessors and logs finish reasons.
        """
        # Log finish reason if available
        try:
            finish_reason = getattr(response, "finish_reason", None)
            if not finish_reason and getattr(response, "candidates", None):
                first_cand = response.candidates[0]
                finish_reason = (
                    getattr(first_cand, "finish_reason", None) or 
                    getattr(first_cand, "metadata", {}).get("finish_reason")
                )
            if finish_reason is not None:
                logger.debug(f"Gemini response finish_reason: {finish_reason}")
        except Exception:
            pass
        
        # 1) Try quick .text accessor
        try:
            text = getattr(response, "text", None)
            if text:
                return text
        except Exception:
            pass
        
        # 2) Try candidates -> content -> parts -> text
        try:
            if getattr(response, "candidates", None):
                parts = []
                for cand in response.candidates:
                    # Try candidate.text
                    if getattr(cand, "text", None):
                        parts.append(cand.text)
                        continue
                    
                    # Try candidate.content (list of parts)
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
        
        # 3) Try output -> content -> text
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
        
        # 4) Last resort: try to stringify
        try:
            if hasattr(response, "to_dict"):
                return json.dumps(response.to_dict(), indent=2)
        except Exception:
            pass
        
        logger.warning("Could not extract text from Gemini response")
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
                print(f"✓ Gemini API connection successful")
                print(f"Test response: {text}")
                return True
            else:
                logger.error("✗ Gemini API connection failed: no text returned")
                print("✗ Gemini API connection failed: no text returned")
                return False
                
        except Exception as e:
            logger.error(f"✗ Gemini API connection failed: {e}")
            print(f"✗ Gemini API connection failed: {e}")
            return False