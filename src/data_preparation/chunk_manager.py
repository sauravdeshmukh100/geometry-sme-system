import re
from typing import List, Dict, Any, Tuple, Optional
import tiktoken
import logging
from dataclasses import dataclass, field
import uuid

from ..config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    chunk_id: str
    text: str
    level: int  # 0, 1, or 2
    parent_id: Optional[str] = None
    doc_id: Optional[str] = None
    source: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class GeometryChunkManager:
    """Manages hierarchical chunking for geometry content."""
    
    def __init__(self):
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoder: {e}. Using word-count fallback.")
            self.tokenizer = None
            
        self.chunk_sizes = [
            settings.chunk_size_level_0,  # 2048 tokens
            settings.chunk_size_level_1,  # 512 tokens
            settings.chunk_size_level_2   # 128 tokens
        ]
        self.overlap_tokens = settings.chunk_overlap
        
    def create_hierarchical_chunks(
        self, 
        text: str, 
        doc_id: str, 
        source: str,
        doc_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Create hierarchical chunks from text.
        
        Args:
            text: Source text to chunk
            doc_id: Document identifier
            source: Source file name
            doc_metadata: Metadata from document (grade_level, difficulty, etc.)
        """
        all_chunks = []
        
        if not text or not text.strip():
            logger.warning(f"Empty text received for {source}")
            return all_chunks
        
        # Level 0: Large context chunks
        level_0_chunks = self._create_chunks(
            text, 
            self.chunk_sizes[0], 
            level=0,
            doc_id=doc_id,
            source=source,
            doc_metadata=doc_metadata
        )
        all_chunks.extend(level_0_chunks)
        
        logger.info(f"Created {len(level_0_chunks)} Level-0 chunks for {source}")
        
        # Level 1: Medium detail chunks
        for l0_chunk in level_0_chunks:
            level_1_chunks = self._create_chunks(
                l0_chunk.text,
                self.chunk_sizes[1],
                level=1,
                parent_id=l0_chunk.chunk_id,
                doc_id=doc_id,
                source=source,
                base_char_offset=l0_chunk.start_char,
                doc_metadata=doc_metadata
            )
            all_chunks.extend(level_1_chunks)
            
            # Level 2: Fine-grained chunks
            for l1_chunk in level_1_chunks:
                level_2_chunks = self._create_chunks(
                    l1_chunk.text,
                    self.chunk_sizes[2],
                    level=2,
                    parent_id=l1_chunk.chunk_id,
                    doc_id=doc_id,
                    source=source,
                    base_char_offset=l1_chunk.start_char,
                    doc_metadata=doc_metadata
                )
                all_chunks.extend(level_2_chunks)
        
        # Add geometry-specific metadata to chunks
        self._add_geometry_metadata(all_chunks, doc_metadata)
        
        logger.info(f"Total chunks created for {source}: {len(all_chunks)} "
                   f"(L0: {len(level_0_chunks)}, "
                   f"L1: {sum(1 for c in all_chunks if c.level == 1)}, "
                   f"L2: {sum(1 for c in all_chunks if c.level == 2)})")
        
        return all_chunks
    
    def _create_chunks(
        self,
        text: str,
        max_tokens: int,
        level: int,
        doc_id: str,
        source: str,
        parent_id: Optional[str] = None,
        base_char_offset: int = 0,
        doc_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Create chunks of specified size with sentence-based segmentation and overlap."""
        chunks = []
        
        if not text or not text.strip():
            return chunks

        # Try splitting into sentences
        sentences = self._split_into_sentences(text)
        
        # Fallback if sentence splitting fails
        if not sentences or all(len(s.strip()) == 0 for s in sentences):
            logger.warning(f"[Chunking] No valid sentences detected in {source}. "
                          f"Using paragraph split fallback.")
            sentences = [p.strip() for p in re.split(r'\n{2,}|\n', text) if p.strip()]
            
            # If still nothing, split on periods as last resort
            if not sentences:
                sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]

        current_chunk = []
        current_tokens = 0
        current_start_char = 0

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_tokens = self._count_tokens(sentence)

            # If one sentence itself is longer than max_tokens, split it
            if sentence_tokens > max_tokens:
                logger.debug(f"[Chunking] Sentence exceeds max_tokens ({sentence_tokens} > {max_tokens}). "
                           f"Splitting sentence.")
                # Split long sentence into smaller parts
                words = sentence.split()
                mid_point = len(words) // 2
                
                first_half = ' '.join(words[:mid_point])
                second_half = ' '.join(words[mid_point:])
                
                # Insert the split parts back
                sentences[i] = first_half
                sentences.insert(i + 1, second_half)
                
                # Reprocess this sentence
                sentence = first_half
                sentence_tokens = self._count_tokens(sentence)

            # Try to add sentence to current chunk
            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunk_id = self._generate_chunk_id()

                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        level=level,
                        parent_id=parent_id,
                        doc_id=doc_id,
                        source=source,
                        start_char=base_char_offset + current_start_char,
                        end_char=base_char_offset + current_start_char + len(chunk_text),
                        metadata={}
                    ))

                    # Handle overlap
                    if self.overlap_tokens > 0 and len(current_chunk) > 1:
                        overlap_sentences = []
                        overlap_tokens = 0
                        
                        for sent in reversed(current_chunk):
                            sent_tokens = self._count_tokens(sent)
                            if overlap_tokens + sent_tokens <= self.overlap_tokens:
                                overlap_sentences.insert(0, sent)
                                overlap_tokens += sent_tokens
                            else:
                                break
                        
                        current_chunk = overlap_sentences + [sentence]
                        current_tokens = overlap_tokens + sentence_tokens
                    else:
                        current_chunk = [sentence]
                        current_tokens = sentence_tokens

                    current_start_char += len(chunk_text) + 1
                else:
                    # Edge case: first sentence itself is problematic
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens

        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = self._generate_chunk_id()
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                level=level,
                parent_id=parent_id,
                doc_id=doc_id,
                source=source,
                start_char=base_char_offset + current_start_char,
                end_char=base_char_offset + current_start_char + len(chunk_text),
                metadata={}
            ))

        # Fallback: If still no chunks created, create one chunk for entire text
        if not chunks and text.strip():
            logger.warning(f"[Chunking] Fallback: wrapping full text of {source} "
                          f"(level {level}) into one chunk.")
            chunk_id = self._generate_chunk_id()
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=text.strip(),
                level=level,
                parent_id=parent_id,
                doc_id=doc_id,
                source=source,
                start_char=base_char_offset,
                end_char=base_char_offset + len(text),
                metadata={}
            ))

        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with geometry-aware rules."""
        if not text:
            return []
            
        # Protect mathematical expressions and formulas
        text = re.sub(r'(\d+)\.(\d+)', r'\1[DOT]\2', text)  # Protect decimals
        text = re.sub(r'([A-Z])\.([A-Z])', r'\1[DOT]\2', text)  # Protect abbreviations
        
        # Split on sentence boundaries
        # Look for period, exclamation, or question mark followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore protected patterns
        sentences = [s.replace('[DOT]', '.') for s in sentences]
        
        # Further split very long sentences
        final_sentences = []
        for sentence in sentences:
            if len(sentence) > 500:  # Long sentence threshold
                # Try to split on semicolons or commas
                parts = re.split(r'[;]\s+', sentence)
                if len(parts) > 1:
                    final_sentences.extend(parts)
                else:
                    # If no semicolons, split on commas as last resort
                    parts = re.split(r'[,]\s+', sentence)
                    # But only split if we get reasonable-sized parts
                    if all(len(p) > 50 for p in parts):
                        final_sentences.extend(parts)
                    else:
                        final_sentences.append(sentence)
            else:
                final_sentences.append(sentence)
        
        return [s.strip() for s in final_sentences if s.strip()]
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken, with fallback."""
        if not text:
            return 0
            
        try:
            if self.tokenizer:
                tokens = self.tokenizer.encode(text)
                return len(tokens) if tokens else max(len(text.split()), 1)
            else:
                # Fallback: approximate tokens as words * 1.3
                return max(int(len(text.split()) * 1.3), 1)
        except Exception as e:
            logger.debug(f"Token counting failed: {e}. Using word count fallback.")
            # Simple word count fallback with adjustment
            return max(int(len(text.split()) * 1.3), 1)
    
    def _generate_chunk_id(self) -> str:
        """Generate unique chunk ID."""
        return str(uuid.uuid4())
    
    def _add_geometry_metadata(
        self, 
        chunks: List[Chunk],
        doc_metadata: Optional[Dict[str, Any]] = None
    ):
        """Add geometry-specific metadata to chunks."""
        for chunk in chunks:
            # Geometry content detection
            metadata = {
                'contains_theorem': bool(re.search(
                    r'\b(theorem|proof|lemma|corollary|proposition)\b', 
                    chunk.text, 
                    re.I
                )),
                'contains_formula': bool(re.search(
                    r'[A-Z]\s*=\s*|[a-z]\s*=\s*\d+|area\s*=|perimeter\s*=', 
                    chunk.text
                )),
                'contains_shape': bool(re.search(
                    r'\b(triangle|square|rectangle|circle|polygon|quadrilateral|'
                    r'pentagon|hexagon|octagon|parallelogram|rhombus|trapezoid)\b', 
                    chunk.text, 
                    re.I
                )),
                'contains_angle': bool(re.search(
                    r'\b(angle|degree|radian|acute|obtuse|right angle|'
                    r'complementary|supplementary)\b', 
                    chunk.text, 
                    re.I
                )),
                'has_numbers': bool(re.search(r'\d+', chunk.text)),
                'topic_density': self._calculate_topic_density(chunk.text),
                'chunk_length': len(chunk.text),
                'word_count': len(chunk.text.split())
            }
            
            # Add document-level metadata if available
            if doc_metadata:
                metadata['grade_level'] = doc_metadata.get('grade_level', 'Unknown')
                metadata['difficulty'] = doc_metadata.get('difficulty', 'Unknown')
                metadata['source_type'] = doc_metadata.get('source_type', 'Unknown')
                
                # Add chapter info if available
                if 'chapter_info' in doc_metadata and doc_metadata['chapter_info']:
                    metadata['chapter_info'] = doc_metadata['chapter_info']
            
            chunk.metadata.update(metadata)
    
    def _calculate_topic_density(self, text: str) -> float:
        """Calculate geometry topic density in text."""
        geometry_terms = [
            'angle', 'triangle', 'square', 'circle', 'polygon', 'theorem',
            'proof', 'congruent', 'similar', 'parallel', 'perpendicular', 
            'area', 'volume', 'perimeter', 'circumference', 'radius', 'diameter',
            'vertex', 'edge', 'face', 'line', 'point', 'plane', 'segment'
        ]
        
        text_lower = text.lower()
        word_count = len(text.split())
        
        if word_count == 0:
            return 0.0
        
        # Count unique geometry terms present
        term_count = sum(1 for term in geometry_terms if term in text_lower)
        
        # Normalize by word count
        density = term_count / max(word_count, 1)
        
        return min(density, 1.0)
    
    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about created chunks."""
        if not chunks:
            return {}
        
        stats = {
            'total_chunks': len(chunks),
            'level_0': sum(1 for c in chunks if c.level == 0),
            'level_1': sum(1 for c in chunks if c.level == 1),
            'level_2': sum(1 for c in chunks if c.level == 2),
            'avg_chunk_length': sum(len(c.text) for c in chunks) / len(chunks),
            'avg_tokens': sum(self._count_tokens(c.text) for c in chunks) / len(chunks),
            'contains_theorem': sum(1 for c in chunks if c.metadata.get('contains_theorem', False)),
            'contains_formula': sum(1 for c in chunks if c.metadata.get('contains_formula', False)),
            'contains_shape': sum(1 for c in chunks if c.metadata.get('contains_shape', False))
        }
        
        return stats