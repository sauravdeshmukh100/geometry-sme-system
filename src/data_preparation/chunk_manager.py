import re
from typing import List, Dict, Any, Tuple
import tiktoken
import logging
from dataclasses import dataclass
import uuid

from ..config.settings import settings

# Logging setup
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    chunk_id: str
    text: str
    level: int  # 0, 1, or 2
    parent_id: str = None
    doc_id: str = None
    source: str = None
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = None

class GeometryChunkManager:
    """Manages hierarchical chunking for geometry content."""
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunk_sizes = [
            settings.chunk_size_level_0,  # 2048 tokens
            settings.chunk_size_level_1,  # 512 tokens
            settings.chunk_size_level_2   # 128 tokens
        ]
        self.overlap_tokens = settings.chunk_overlap
        
    def create_hierarchical_chunks(self, text: str, doc_id: str, source: str) -> List[Chunk]:
        """Create hierarchical chunks from text."""
        all_chunks = []
        
        # Level 0: Large context chunks
        level_0_chunks = self._create_chunks(
            text, 
            self.chunk_sizes[0], 
            level=0,
            doc_id=doc_id,
            source=source
        )
        all_chunks.extend(level_0_chunks)
        
        # Level 1: Medium detail chunks
        for l0_chunk in level_0_chunks:
            level_1_chunks = self._create_chunks(
                l0_chunk.text,
                self.chunk_sizes[1],
                level=1,
                parent_id=l0_chunk.chunk_id,
                doc_id=doc_id,
                source=source,
                base_char_offset=l0_chunk.start_char
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
                    base_char_offset=l1_chunk.start_char
                )
                all_chunks.extend(level_2_chunks)
        
        # Add geometry-specific metadata to chunks
        self._add_geometry_metadata(all_chunks)
        
        return all_chunks
    
    def _create_chunks(
    self,
    text: str,
    max_tokens: int,
    level: int,
    doc_id: str,
    source: str,
    parent_id: str = None,
    base_char_offset: int = 0
) -> List[Chunk]:
        """Create chunks of specified size with sentence-based segmentation and overlap."""
        chunks = []

        # Try splitting into sentences
        sentences = self._split_into_sentences(text)
        if not sentences or all(len(s.strip()) == 0 for s in sentences):
            import re
            logger.warning(f"[Chunking] No valid sentences detected in {source}. Using paragraph split fallback.")
            sentences = [p.strip() for p in re.split(r'\n{2,}|\n|\. ', text) if p.strip()]

        current_chunk = []
        current_tokens = 0
        current_start_char = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # If one sentence itself is longer than max_tokens, break it in half safely
            if sentence_tokens > max_tokens:
                halfway = len(sentence) // 2
                sentences.insert(sentences.index(sentence) + 1, sentence[halfway:])
                sentence = sentence[:halfway]
                sentence_tokens = self._count_tokens(sentence)

            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                # Save current chunk
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

        # 🚨 If still no chunks created, create one fallback chunk for entire text
        if not chunks and text.strip():
            logger.warning(f"[Chunking] Fallback: wrapping full text of {source} into one chunk.")
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
        # Protect mathematical expressions and formulas
        text = re.sub(r'(\d+)\.(\d+)', r'\1[DOT]\2', text)  # Protect decimals
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore protected patterns
        sentences = [s.replace('[DOT]', '.') for s in sentences]
        
        # Further split very long sentences
        final_sentences = []
        for sentence in sentences:
            if len(sentence) > 500:  # Long sentence
                # Try to split on semicolons or commas
                parts = re.split(r'[;,]\s+', sentence)
                final_sentences.extend(parts)
            else:
                final_sentences.append(sentence)
        
        return [s.strip() for s in final_sentences if s.strip()]
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken, with fallback."""
        try:
            tokens = self.tokenizer.encode(text)
            if len(tokens) == 0:
                # fallback for non-English or symbol-heavy text
                return max(len(text.split()), 1)
            return len(tokens)
        except Exception:
            # simple word count fallback
            return max(len(text.split()), 1)

    
    def _generate_chunk_id(self) -> str:
        """Generate unique chunk ID."""
        return str(uuid.uuid4())
    
    def _add_geometry_metadata(self, chunks: List[Chunk]):
        """Add geometry-specific metadata to chunks."""
        for chunk in chunks:
            metadata = {
                'contains_theorem': bool(re.search(r'\b(theorem|proof|lemma)\b', chunk.text, re.I)),
                'contains_formula': bool(re.search(r'[A-Z]\s*=\s*', chunk.text)),
                'contains_shape': bool(re.search(r'\b(triangle|square|circle|polygon)\b', chunk.text, re.I)),
                'contains_angle': bool(re.search(r'\b(angle|degree|radian)\b', chunk.text, re.I)),
                'has_numbers': bool(re.search(r'\d+', chunk.text)),
                'topic_density': self._calculate_topic_density(chunk.text)
            }
            chunk.metadata.update(metadata)
    
    def _calculate_topic_density(self, text: str) -> float:
        """Calculate geometry topic density in text."""
        geometry_terms = [
            'angle', 'triangle', 'square', 'circle', 'polygon', 'theorem',
            'proof', 'congruent', 'parallel', 'perpendicular', 'area', 'volume'
        ]
        
        text_lower = text.lower()
        word_count = len(text.split())
        
        if word_count == 0:
            return 0.0
        
        term_count = sum(1 for term in geometry_terms if term in text_lower)
        return min(term_count / word_count, 1.0)