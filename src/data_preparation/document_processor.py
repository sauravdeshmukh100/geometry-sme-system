import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

import pypdf
from docx import Document as DocxDocument
from pptx import Presentation
import markdown
from bs4 import BeautifulSoup
import re

from ..config.settings import settings

logger = logging.getLogger(__name__)

class GeometryDocumentProcessor:
    """Process various document formats for geometry content."""
    
    SUPPORTED_FORMATS = {'.pdf', '.docx', '.pptx', '.txt', '.md', '.html'}
    
    # Geometry-specific patterns for content identification
    GEOMETRY_PATTERNS = {
        'theorem': r'\b(theorem|proof|lemma|corollary|proposition)\b',
        'shape': r'\b(triangle|square|rectangle|circle|polygon|quadrilateral)\b',
        'angle': r'\b(angle|degree|radian|acute|obtuse|right angle)\b',
        'formula': r'\b(area|perimeter|volume|circumference|pythagorean)\b',
        'concept': r'\b(congruent|similar|parallel|perpendicular|bisect)\b'
    }
    
    def __init__(self):
        self.processed_docs = []
        
    def process_document(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single document and extract geometry content."""
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
            
        file_ext = file_path.suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            logger.warning(f"Unsupported format: {file_ext}")
            return None
            
        try:
            # Extract text based on file type
            if file_ext == '.pdf':
                text = self._extract_pdf(file_path)
            elif file_ext == '.docx':
                text = self._extract_docx(file_path)
            elif file_ext == '.pptx':
                text = self._extract_pptx(file_path)
            elif file_ext == '.md':
                text = self._extract_markdown(file_path)
            elif file_ext == '.html':
                text = self._extract_html(file_path)
            else:  # .txt
                text = self._extract_text(file_path)
            
            # Clean and normalize text
            text = self._clean_text(text)
            
            # Extract geometry-specific metadata
            metadata = self._extract_geometry_metadata(text, file_path)
            
            # Create document record
            doc_record = {
                'doc_id': self._generate_doc_id(file_path),
                'file_name': file_path.name,
                'file_path': str(file_path),
                'content': text,
                'metadata': metadata,
                'processed_at': datetime.now().isoformat(),
                'char_count': len(text),
                'word_count': len(text.split())
            }
            
            # Save processed document
            self._save_processed_document(doc_record)
            
            return doc_record
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        text_parts = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num}: {e}")
                    
        return '\n\n'.join(text_parts)
    
    def _extract_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        doc = DocxDocument(file_path)
        
        text_parts = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
                
        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = ' | '.join(cell.text.strip() for cell in row.cells)
                if row_text.strip():
                    text_parts.append(row_text)
                    
        return '\n\n'.join(text_parts)
    
    def _extract_pptx(self, file_path: Path) -> str:
        """Extract text from PPTX file."""
        prs = Presentation(file_path)
        
        text_parts = []
        for slide_num, slide in enumerate(prs.slides):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, 'text'):
                    if shape.text.strip():
                        slide_text.append(shape.text)
                        
            if slide_text:
                text_parts.append(f"[Slide {slide_num + 1}]\n" + '\n'.join(slide_text))
                
        return '\n\n'.join(text_parts)
    
    def _extract_markdown(self, file_path: Path) -> str:
        """Extract text from Markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            md_text = f.read()
            
        # Convert markdown to HTML then extract text
        html = markdown.markdown(md_text)
        soup = BeautifulSoup(html, 'html.parser')
        
        return soup.get_text()
    
    def _extract_html(self, file_path: Path) -> str:
        """Extract text from HTML file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style']):
            element.decompose()
            
        return soup.get_text()
    
    def _extract_text(self, file_path: Path) -> str:
        """Extract text from plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep geometry symbols
        text = re.sub(r'[^\w\s\-+×÷=°∠∆∏∑√∞≈≠≤≥±∓∝∟⊥∥∦∴∵]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _extract_geometry_metadata(self, text: str, file_path: Path) -> Dict[str, Any]:
        """Extract geometry-specific metadata from text."""
        metadata = {
            'source_file': file_path.name,
            'topics': [],
            'concepts': {},
            'grade_level': self._infer_grade_level(text),
            'difficulty': self._assess_difficulty(text)
        }
        
        # Identify geometry topics
        for topic, pattern in self.GEOMETRY_PATTERNS.items():
            matches = re.findall(pattern, text.lower())
            if matches:
                metadata['concepts'][topic] = list(set(matches))[:10]  # Top 10 unique
                metadata['topics'].append(topic)
        
        # Extract formulas (simple pattern)
        formula_pattern = r'[A-Z]\s*=\s*[^.]+(?:\.|$)'
        formulas = re.findall(formula_pattern, text)
        if formulas:
            metadata['formulas'] = formulas[:5]  # Top 5 formulas
        
        return metadata
    
    def _infer_grade_level(self, text: str) -> str:
        """Infer grade level from content complexity."""
        text_lower = text.lower()
        
        # Simple heuristics for grade level
        if any(term in text_lower for term in ['basic shapes', 'counting sides', 'simple angles']):
            return 'Elementary (K-5)'
        elif any(term in text_lower for term in ['pythagorean', 'coordinate plane', 'transformations']):
            return 'Middle School (6-8)'
        elif any(term in text_lower for term in ['trigonometry', 'proofs', 'theorems', 'calculus']):
            return 'High School (9-12)'
        else:
            return 'General'
    
    def _assess_difficulty(self, text: str) -> str:
        """Assess content difficulty level."""
        text_lower = text.lower()
        
        # Count complexity indicators
        complex_terms = ['proof', 'theorem', 'lemma', 'corollary', 'axiom', 'postulate']
        complexity_score = sum(1 for term in complex_terms if term in text_lower)
        
        if complexity_score >= 4:
            return 'Advanced'
        elif complexity_score >= 2:
            return 'Intermediate'
        else:
            return 'Beginner'
    
    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate unique document ID."""
        content = f"{file_path.name}{file_path.stat().st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _save_processed_document(self, doc_record: Dict[str, Any]):
        """Save processed document to file."""
        output_path = settings.processed_data_dir / f"{doc_record['doc_id']}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_record, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved processed document: {doc_record['file_name']}")
    
    def process_directory(self, directory: Path) -> List[Dict[str, Any]]:
        """Process all documents in a directory."""
        processed_docs = []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                logger.info(f"Processing: {file_path.name}")
                doc_record = self.process_document(file_path)
                
                if doc_record:
                    processed_docs.append(doc_record)
                    
        logger.info(f"Processed {len(processed_docs)} documents")
        return processed_docs