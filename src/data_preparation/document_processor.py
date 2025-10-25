import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import re

import pypdf
from docx import Document as DocxDocument
from pptx import Presentation
import markdown
from bs4 import BeautifulSoup

from ..config.settings import settings

logger = logging.getLogger(__name__)

class GeometryDocumentProcessor:
    """Process various document formats for geometry content with grade-specific handling."""
    
    SUPPORTED_FORMATS = {'.pdf', '.docx', '.pptx', '.txt', '.md', '.html'}
    
    # Geometry-specific patterns for content identification
    GEOMETRY_PATTERNS = {
        'theorem': r'\b(theorem|proof|lemma|corollary|proposition)\b',
        'shape': r'\b(triangle|square|rectangle|circle|polygon|quadrilateral|pentagon|hexagon|octagon)\b',
        'angle': r'\b(angle|degree|radian|acute|obtuse|right angle|complementary|supplementary)\b',
        'formula': r'\b(area|perimeter|volume|circumference|pythagorean|surface area)\b',
        'concept': r'\b(congruent|similar|parallel|perpendicular|bisect|symmetry|transformation)\b'
    }
    
    # Grade-specific topic keywords for better classification
    GRADE_LEVEL_KEYWORDS = {
        'grade_6': ['basic shapes', 'perimeter', 'area of rectangle', 'area of square', 
                    'circle basics', 'simple angles', 'line segment', 'ray'],
        'grade_7': ['triangles', 'congruence', 'construction', 'symmetry', 
                    'area of parallelogram', 'area of triangle'],
        'grade_8': ['quadrilaterals', 'mensuration', 'volume', 'surface area',
                    'understanding quadrilaterals', 'visualizing solid shapes'],
        'grade_9': ['coordinate geometry', 'lines and angles', 'triangles advanced',
                    'heron formula', 'areas of parallelograms', 'circles'],
        'grade_10': ['similar triangles', 'pythagoras theorem', 'trigonometry',
                     'circles advanced', 'tangent', 'constructions', 'areas related to circles']
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
            # Extract grade from filename first (for NCERT files)
            grade_from_filename = self._extract_grade_from_filename(file_path.name)
            
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
            
            # Determine source type
            source_type = self._identify_source_type(file_path.name, text)
            
            # Extract geometry-specific metadata
            metadata = self._extract_geometry_metadata(
                text, 
                file_path, 
                grade_from_filename,
                source_type
            )
            
            # Create document record
            doc_record = {
                'doc_id': self._generate_doc_id(file_path),
                'file_name': file_path.name,
                'file_path': str(file_path),
                'content': text,
                'metadata': metadata,
                'processed_at': datetime.now().isoformat(),
                'char_count': len(text),
                'word_count': len(text.split()),
                'source_type': source_type
            }
            
            # Save processed document
            self._save_processed_document(doc_record)
            
            return doc_record
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def _extract_grade_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract grade level from filename.
        Handles patterns like: class6_9.pdf, grade7_chapter3.pdf, etc.
        """
        filename_lower = filename.lower()
        
        # Pattern 1: classX_Y.pdf or gradeX_Y.pdf
        match = re.search(r'(?:class|grade)(\d+)(?:_|-)?\d*', filename_lower)
        if match:
            grade_num = int(match.group(1))
            if 6 <= grade_num <= 10:
                return f"Grade {grade_num}"
        
        # Pattern 2: Just number like 6.pdf, 7.pdf
        match = re.search(r'^(\d+)(?:_|\.).*\.pdf$', filename_lower)
        if match:
            grade_num = int(match.group(1))
            if 6 <= grade_num <= 10:
                return f"Grade {grade_num}"
        
        return None
    
    def _identify_source_type(self, filename: str, text: str) -> str:
        """Identify the type of source document."""
        filename_lower = filename.lower()
        text_lower = text.lower()
        
        # NCERT identification
        if 'class' in filename_lower and any(f'class{i}' in filename_lower for i in range(6, 11)):
            return 'NCERT Textbook'
        
        # General textbook identification
        if 'textbook' in filename_lower or 'geometry for enjoyment' in text_lower:
            return 'Textbook'
        
        # PPT identification
        if filename.endswith(('.ppt', '.pptx')):
            return 'Presentation'
        
        # Default
        return 'Supplementary Material'
    
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
        """Extract text from PPTX file with slide titles as context."""
        prs = Presentation(file_path)
        
        text_parts = []
        # Add presentation title from filename as context
        ppt_title = file_path.stem.replace('_', ' ').replace('-', ' ').title()
        text_parts.append(f"[Presentation: {ppt_title}]")
        
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
        # Keep common math symbols: +, -, ×, ÷, =, °, ∠, ∆, π, √, ∞, ≈, ≠, ≤, ≥, ±, ⊥, ∥
        text = re.sub(r'[^\w\s\-+×÷=°∠∆∏∑√∞≈≠≤≥±∓∝∟⊥∥∦∴∵\(\)\[\]\{\}\/\.\,\:\;]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _extract_geometry_metadata(
        self, 
        text: str, 
        file_path: Path,
        grade_from_filename: Optional[str],
        source_type: str
    ) -> Dict[str, Any]:
        """Extract geometry-specific metadata from text with enhanced grade detection."""
        
        # Determine grade level (prioritize filename, then content analysis)
        if grade_from_filename:
            grade_level = grade_from_filename
        else:
            grade_level = self._infer_grade_level(text, source_type)
        
        metadata = {
            'source_file': file_path.name,
            'source_type': source_type,
            'topics': [],
            'concepts': {},
            'grade_level': grade_level,
            'difficulty': self._assess_difficulty(text, grade_level),
            'chapter_info': self._extract_chapter_info(file_path.name)
        }
        
        # Identify geometry topics
        for topic, pattern in self.GEOMETRY_PATTERNS.items():
            matches = re.findall(pattern, text.lower())
            if matches:
                metadata['concepts'][topic] = list(set(matches))[:10]  # Top 10 unique
                metadata['topics'].append(topic)
        
        # Extract formulas (enhanced pattern)
        formula_pattern = r'[A-Z]\s*=\s*[^.]+(?:\.|$)'
        formulas = re.findall(formula_pattern, text)
        if formulas:
            metadata['formulas'] = formulas[:5]  # Top 5 formulas
        
        return metadata
    
    def _extract_chapter_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Extract chapter information from filename."""
        # Pattern for class6_9.pdf (class 6, chapter 9)
        match = re.search(r'(?:class|grade)(\d+)(?:_|-)(\d+)', filename.lower())
        if match:
            return {
                'grade': int(match.group(1)),
                'chapter': int(match.group(2))
            }
        return None
    
    def _infer_grade_level(self, text: str, source_type: str) -> str:
        """
        Infer grade level from content complexity with enhanced logic.
        Focus on grades 6-10 as per your requirement.
        """
        text_lower = text.lower()
        
        # For NCERT, try to detect from chapter content
        if source_type == 'NCERT Textbook':
            return self._infer_ncert_grade(text_lower)
        
        # For general textbook (like Geometry for Enjoyment and Challenge)
        # This book covers grades 6-12, so we analyze content complexity
        if source_type == 'Textbook':
            return self._infer_textbook_grade(text_lower)
        
        # For presentations, use keywords
        if source_type == 'Presentation':
            return self._infer_grade_from_keywords(text_lower)
        
        # Default fallback
        return self._infer_grade_from_keywords(text_lower)
    
    def _infer_ncert_grade(self, text_lower: str) -> str:
        """Infer NCERT grade level from content."""
        grade_scores = {f'Grade {i}': 0 for i in range(6, 11)}
        
        # Score each grade based on keyword presence
        for grade, keywords in self.GRADE_LEVEL_KEYWORDS.items():
            grade_num = grade.split('_')[1]
            grade_label = f'Grade {grade_num}'
            for keyword in keywords:
                if keyword in text_lower:
                    grade_scores[grade_label] += 1
        
        # Return grade with highest score
        max_grade = max(grade_scores, key=grade_scores.get)
        if grade_scores[max_grade] > 0:
            return max_grade
        
        return 'Grade 6-10 (General)'
    
    def _infer_textbook_grade(self, text_lower: str) -> str:
        """
        Infer grade for comprehensive textbooks.
        For 'Geometry for Enjoyment and Challenge', map chapters to grades.
        """
        # Basic concepts → Grade 6-7
        if any(term in text_lower for term in [
            'points and lines', 'planes', 'segments', 'rays',
            'basic constructions', 'measuring angles'
        ]):
            return 'Grade 6-7'
        
        # Intermediate concepts → Grade 8-9
        elif any(term in text_lower for term in [
            'congruent triangles', 'quadrilaterals', 'parallel lines',
            'area formulas', 'volume', 'surface area'
        ]):
            return 'Grade 8-9'
        
        # Advanced concepts → Grade 10+
        elif any(term in text_lower for term in [
            'coordinate geometry', 'trigonometry', 'similar triangles',
            'circles and tangents', 'proofs of theorems'
        ]):
            return 'Grade 10'
        
        # Very advanced → Grade 11-12 (but we'll mark as Grade 10 since we focus on 6-10)
        elif any(term in text_lower for term in [
            'analytic geometry', 'conic sections', 'vectors',
            'transformations', 'advanced trigonometry'
        ]):
            return 'Grade 10 (Advanced)'
        
        return 'Grade 6-10 (General)'
    
    def _infer_grade_from_keywords(self, text_lower: str) -> str:
        """Infer grade from keyword analysis."""
        # Count matches for each grade
        grade_scores = {}
        
        for grade_key, keywords in self.GRADE_LEVEL_KEYWORDS.items():
            grade_num = grade_key.split('_')[1]
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                grade_scores[f'Grade {grade_num}'] = score
        
        if grade_scores:
            # Return grade with most keyword matches
            return max(grade_scores, key=grade_scores.get)
        
        # Fallback based on complexity
        complex_terms = ['proof', 'theorem', 'lemma', 'corollary', 'axiom', 'postulate']
        complexity_score = sum(1 for term in complex_terms if term in text_lower)
        
        if complexity_score >= 4:
            return 'Grade 10'
        elif complexity_score >= 2:
            return 'Grade 8-9'
        else:
            return 'Grade 6-7'
    
    def _assess_difficulty(self, text: str, grade_level: str) -> str:
        """Assess content difficulty level based on grade and content."""
        text_lower = text.lower()
        
        # Count complexity indicators
        complex_terms = ['proof', 'theorem', 'lemma', 'corollary', 'axiom', 
                        'postulate', 'derive', 'prove']
        complexity_score = sum(1 for term in complex_terms if term in text_lower)
        
        # Adjust difficulty based on grade
        if 'Grade 6' in grade_level or 'Grade 7' in grade_level:
            if complexity_score >= 2:
                return 'Intermediate'
            else:
                return 'Beginner'
        
        elif 'Grade 8' in grade_level or 'Grade 9' in grade_level:
            if complexity_score >= 4:
                return 'Advanced'
            elif complexity_score >= 2:
                return 'Intermediate'
            else:
                return 'Beginner'
        
        elif 'Grade 10' in grade_level:
            if complexity_score >= 5:
                return 'Advanced'
            elif complexity_score >= 2:
                return 'Intermediate'
            else:
                return 'Beginner'
        
        else:
            # General case
            if complexity_score >= 4:
                return 'Advanced'
            elif complexity_score >= 2:
                return 'Intermediate'
            else:
                return 'Beginner'
    
    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate unique document ID."""
        content = f"{file_path.name}{file_path.stat().st_size}{file_path.stat().st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _save_processed_document(self, doc_record: Dict[str, Any]):
        """Save processed document to file."""
        output_path = settings.processed_data_dir / f"{doc_record['doc_id']}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_record, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved processed document: {doc_record['file_name']} "
                   f"[{doc_record['metadata']['grade_level']}]")
    
    def process_directory(self, directory: Path) -> List[Dict[str, Any]]:
        """Process all documents in a directory with progress reporting."""
        processed_docs = []
        
        # Get all files first
        all_files = [f for f in directory.rglob('*') 
                     if f.is_file() and f.suffix.lower() in self.SUPPORTED_FORMATS]
        
        logger.info(f"Found {len(all_files)} documents to process")
        
        for file_path in all_files:
            logger.info(f"Processing: {file_path.name}")
            doc_record = self.process_document(file_path)
            
            if doc_record:
                processed_docs.append(doc_record)
                logger.info(f"  ✓ Grade: {doc_record['metadata']['grade_level']}, "
                          f"Difficulty: {doc_record['metadata']['difficulty']}")
            else:
                logger.warning(f"  ✗ Failed to process: {file_path.name}")
                    
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Summary:")
        logger.info(f"  Total processed: {len(processed_docs)}/{len(all_files)}")
        
        # Grade distribution
        grade_dist = {}
        for doc in processed_docs:
            grade = doc['metadata']['grade_level']
            grade_dist[grade] = grade_dist.get(grade, 0) + 1
        
        logger.info(f"\n  Grade Distribution:")
        for grade, count in sorted(grade_dist.items()):
            logger.info(f"    {grade}: {count} documents")
        
        logger.info(f"{'='*60}\n")
        
        return processed_docs