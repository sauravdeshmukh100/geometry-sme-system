import re
from typing import List, Dict, Any, Optional
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
    embeddable: bool = True

class HybridRecursiveChunkManager:
    """
    Improved hierarchical chunking with iterative splitting.
    Avoids recursion errors by using iterative approach.
    """
    
    def __init__(self):
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken: {e}")
            self.tokenizer = None
        
        # UPDATED: Model limit is 384 tokens (not 512)
        self.embedding_model_limit = 384
        
        # Target chunk sizes
        self.target_sizes = {
            0: 2048,  # Context layer (NOT embedded)
            1: 350,   # Medium chunks (target, max 384)
            2: 100    # Fine chunks (target, max 128)
        }
        
        # Hard limits for embedding
        self.max_embeddable_tokens = {
            1: 384,   # Level 1 MUST fit in model
            2: 128    # Level 2 smaller for precision
        }
        
        self.overlap_tokens = settings.CHUNK_OVERLAP
        
        # Separators in order of preference
        self.separators = [
            "\n\n\n",      # Major breaks
            "\n\n",        # Paragraph breaks
            "\n",          # Line breaks
            ". ",          # Sentences
            "? ",
            "! ",
            "; ",
            ", ",
            " "            # Words (last resort)
        ]
    
    def create_hierarchical_chunks(
        self, 
        text: str, 
        doc_id: str, 
        source: str,
        doc_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Create hierarchical chunks with iterative splitting."""
        all_chunks = []
        
        if not text or not text.strip():
            logger.warning(f"Empty text for {source}")
            return all_chunks
        
        try:
            # LEVEL 0: Create large context chunks (NOT embedded)
            logger.info(f"Creating Level 0 context chunks for {source}")
            level_0_chunks = self._create_context_chunks(
                text, doc_id, source, doc_metadata
            )
            all_chunks.extend(level_0_chunks)
            
            # LEVEL 1: Split L0 into embeddable chunks iteratively
            logger.info(f"Creating Level 1 chunks (≤384 tokens)")
            for l0_chunk in level_0_chunks:
                level_1_chunks = self._iterative_split(
                    text=l0_chunk.text,
                    max_tokens=self.max_embeddable_tokens[1],
                    level=1,
                    parent_id=l0_chunk.chunk_id,
                    doc_id=doc_id,
                    source=source,
                    base_char_offset=l0_chunk.start_char,
                    doc_metadata=doc_metadata
                )
                all_chunks.extend(level_1_chunks)
                
                # LEVEL 2: Split L1 into fine-grained chunks
                logger.info(f"Creating Level 2 chunks (≤128 tokens)")
                for l1_chunk in level_1_chunks:
                    level_2_chunks = self._iterative_split(
                        text=l1_chunk.text,
                        max_tokens=self.max_embeddable_tokens[2],
                        level=2,
                        parent_id=l1_chunk.chunk_id,
                        doc_id=doc_id,
                        source=source,
                        base_char_offset=l1_chunk.start_char,
                        doc_metadata=doc_metadata
                    )
                    all_chunks.extend(level_2_chunks)
            
            # Add geometry-specific metadata
            self._add_geometry_metadata(all_chunks, doc_metadata)
            
            logger.info(f"Created {len(all_chunks)} total chunks for {source}")
            
        except Exception as e:
            logger.error(f"Error creating chunks for {source}: {e}", exc_info=True)
            return []
        
        return all_chunks
    
    def _create_context_chunks(
        self,
        text: str,
        doc_id: str,
        source: str,
        doc_metadata: Optional[Dict[str, Any]]
    ) -> List[Chunk]:
        """Create Level 0 context chunks (~2048 tokens)."""
        chunks = []
        target_size = self.target_sizes[0]
        
        # Split on double newlines (paragraphs)
        sections = text.split('\n\n')
        
        current_chunk_text = []
        current_tokens = 0
        current_start = 0
        
        for section in sections:
            if not section.strip():
                continue
                
            section_tokens = self._count_tokens(section)
            
            if current_tokens + section_tokens <= target_size:
                current_chunk_text.append(section)
                current_tokens += section_tokens
            else:
                # Save current chunk
                if current_chunk_text:
                    chunk_text = '\n\n'.join(current_chunk_text)
                    chunks.append(Chunk(
                        chunk_id=self._generate_chunk_id(),
                        text=chunk_text,
                        level=0,
                        doc_id=doc_id,
                        source=source,
                        start_char=current_start,
                        end_char=current_start + len(chunk_text),
                        embeddable=False,
                        metadata={}
                    ))
                    current_start += len(chunk_text)
                
                # Start new chunk
                current_chunk_text = [section]
                current_tokens = section_tokens
        
        # Add remaining
        if current_chunk_text:
            chunk_text = '\n\n'.join(current_chunk_text)
            chunks.append(Chunk(
                chunk_id=self._generate_chunk_id(),
                text=chunk_text,
                level=0,
                doc_id=doc_id,
                source=source,
                start_char=current_start,
                end_char=current_start + len(chunk_text),
                embeddable=False,
                metadata={}
            ))
        
        return chunks if chunks else [Chunk(
            chunk_id=self._generate_chunk_id(),
            text=text,
            level=0,
            doc_id=doc_id,
            source=source,
            start_char=0,
            end_char=len(text),
            embeddable=False,
            metadata={}
        )]
    
    def _iterative_split(
        self,
        text: str,
        max_tokens: int,
        level: int,
        parent_id: str,
        doc_id: str,
        source: str,
        base_char_offset: int,
        doc_metadata: Optional[Dict[str, Any]]
    ) -> List[Chunk]:
        """
        Iteratively split text until all chunks ≤ max_tokens.
        No recursion - uses queue-based approach.
        """
        chunks = []
        
        # Queue of texts to process: (text, start_offset)
        queue = [(text, base_char_offset)]
        
        max_iterations = 1000  # Safety limit
        iteration = 0
        
        while queue and iteration < max_iterations:
            iteration += 1
            current_text, start_offset = queue.pop(0)
            
            # Check if small enough
            token_count = self._count_tokens(current_text)
            
            if token_count <= max_tokens:
                # Small enough - create chunk
                chunks.append(Chunk(
                    chunk_id=self._generate_chunk_id(),
                    text=current_text,
                    level=level,
                    parent_id=parent_id,
                    doc_id=doc_id,
                    source=source,
                    start_char=start_offset,
                    end_char=start_offset + len(current_text),
                    embeddable=True,
                    metadata={}
                ))
            else:
                # Too large - split it
                parts = self._split_text(current_text, max_tokens)
                
                if len(parts) <= 1:
                    # Couldn't split - force split by character count
                    mid = len(current_text) // 2
                    # Find nearest space
                    while mid < len(current_text) and current_text[mid] != ' ':
                        mid += 1
                    
                    if mid < len(current_text):
                        parts = [current_text[:mid], current_text[mid:]]
                    else:
                        # No space found - just create oversized chunk
                        logger.warning(f"Creating oversized chunk ({token_count} tokens) for {source}")
                        chunks.append(Chunk(
                            chunk_id=self._generate_chunk_id(),
                            text=current_text,
                            level=level,
                            parent_id=parent_id,
                            doc_id=doc_id,
                            source=source,
                            start_char=start_offset,
                            end_char=start_offset + len(current_text),
                            embeddable=True,
                            metadata={}
                        ))
                        continue
                
                # Add parts back to queue
                current_offset = start_offset
                for part in parts:
                    if part.strip():
                        queue.append((part, current_offset))
                        current_offset += len(part)
        
        if iteration >= max_iterations:
            logger.warning(f"Hit max iterations for {source}")
        
        return chunks
    
    def _split_text(self, text: str, max_tokens: int) -> List[str]:
        """
        Split text at natural boundaries.
        Returns list of parts, each hopefully ≤ max_tokens.
        """
        # Try each separator in order
        for separator in self.separators:
            if separator in text:
                parts = text.split(separator)
                
                # Check if split is reasonable
                if len(parts) > 1:
                    # Reassemble with separator
                    result = []
                    current = []
                    current_tokens = 0
                    
                    for i, part in enumerate(parts):
                        part_tokens = self._count_tokens(part)
                        
                        if current_tokens + part_tokens <= max_tokens:
                            current.append(part)
                            current_tokens += part_tokens
                        else:
                            if current:
                                result.append(separator.join(current))
                            current = [part]
                            current_tokens = part_tokens
                    
                    if current:
                        result.append(separator.join(current))
                    
                    # Verify split was useful
                    if len(result) > 1:
                        return result
        
        # No good separator found - return as-is
        return [text]
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens with fallback."""
        if not text:
            return 0
        try:
            if self.tokenizer:
                tokens = self.tokenizer.encode(text)
                return len(tokens)
            else:
                # Fallback: words * 1.3
                return max(int(len(text.split()) * 1.3), 1)
        except:
            return max(int(len(text.split()) * 1.3), 1)
    
    def _generate_chunk_id(self) -> str:
        return str(uuid.uuid4())
    
    def _add_geometry_metadata(
        self,
        chunks: List[Chunk],
        doc_metadata: Optional[Dict[str, Any]]
    ):
        """Add geometry-specific metadata."""
        for chunk in chunks:
            metadata = {
                'contains_theorem': bool(re.search(
                    r'\b(theorem|proof|lemma|corollary)\b', 
                    chunk.text, re.I
                )),
                'contains_formula': bool(re.search(
                    r'[A-Z]\s*=\s*', chunk.text
                )),
                'contains_shape': bool(re.search(
                    r'\b(triangle|square|circle|polygon)\b', 
                    chunk.text, re.I
                )),
                'embeddable': chunk.embeddable,
                'token_count': self._count_tokens(chunk.text)
            }
            
            if doc_metadata:
                metadata['grade_level'] = doc_metadata.get('grade_level')
                metadata['difficulty'] = doc_metadata.get('difficulty')
                metadata['source_type'] = doc_metadata.get('source_type')
            
            chunk.metadata.update(metadata)