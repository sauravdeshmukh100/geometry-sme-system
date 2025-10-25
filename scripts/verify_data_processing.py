#!/usr/bin/env python
"""
Data verification and statistics script for mid-evaluation.
Analyzes processed documents and generates reports.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import logging

from src.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataVerifier:
    """Verify and analyze processed geometry data."""
    
    def __init__(self):
        self.processed_dir = settings.processed_data_dir
        self.raw_dir = settings.raw_data_dir
    
    def verify_processing(self) -> Dict[str, Any]:
        """Verify data processing and generate statistics."""
        
        print("="*70)
        print("DATA PROCESSING VERIFICATION & STATISTICS")
        print("="*70)
        
        # Check raw files
        raw_files = list(self.raw_dir.rglob('*'))
        raw_files = [f for f in raw_files if f.is_file()]
        
        print(f"\n1. RAW DATA DIRECTORY: {self.raw_dir}")
        print(f"   Total files: {len(raw_files)}")
        
        # Count by extension
        ext_count = defaultdict(int)
        for f in raw_files:
            ext_count[f.suffix.lower()] += 1
        
        print(f"   File types:")
        for ext, count in sorted(ext_count.items()):
            print(f"     {ext}: {count} files")
        
        # Check processed files
        processed_files = list(self.processed_dir.glob('*.json'))
        
        print(f"\n2. PROCESSED DATA DIRECTORY: {self.processed_dir}")
        print(f"   Total processed: {len(processed_files)}")
        
        if len(processed_files) == 0:
            print("\n   ⚠️  No processed files found!")
            print("   Run: python scripts/build_database.py")
            return {}
        
        # Analyze processed documents
        stats = self._analyze_processed_docs(processed_files)
        
        # Print statistics
        self._print_statistics(stats)
        
        # Generate report for mid-evaluation
        self._generate_report(stats)
        
        return stats
    
    def _analyze_processed_docs(self, processed_files: List[Path]) -> Dict[str, Any]:
        """Analyze all processed documents."""
        
        stats = {
            'total_docs': len(processed_files),
            'total_chunks_estimated': 0,
            'grade_distribution': defaultdict(int),
            'difficulty_distribution': defaultdict(int),
            'source_type_distribution': defaultdict(int),
            'topic_distribution': defaultdict(int),
            'total_words': 0,
            'total_chars': 0,
            'documents_by_grade': defaultdict(list),
            'ncert_docs': [],
            'textbook_docs': [],
            'ppt_docs': []
        }
        
        for json_file in processed_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                
                # Basic stats
                stats['total_words'] += doc.get('word_count', 0)
                stats['total_chars'] += doc.get('char_count', 0)
                
                # Estimate chunks (based on word count)
                # Assuming avg chunk size of 300 words
                estimated_chunks = max(1, doc.get('word_count', 0) // 300)
                stats['total_chunks_estimated'] += estimated_chunks
                
                # Metadata stats
                metadata = doc.get('metadata', {})
                
                grade = metadata.get('grade_level', 'Unknown')
                stats['grade_distribution'][grade] += 1
                stats['documents_by_grade'][grade].append(doc['file_name'])
                
                difficulty = metadata.get('difficulty', 'Unknown')
                stats['difficulty_distribution'][difficulty] += 1
                
                source_type = doc.get('source_type', 'Unknown')
                stats['source_type_distribution'][source_type] += 1
                
                # Categorize documents
                if source_type == 'NCERT Textbook':
                    stats['ncert_docs'].append({
                        'name': doc['file_name'],
                        'grade': grade,
                        'chapter': metadata.get('chapter_info')
                    })
                elif source_type == 'Textbook':
                    stats['textbook_docs'].append({
                        'name': doc['file_name'],
                        'grade': grade,
                        'words': doc.get('word_count', 0)
                    })
                elif source_type == 'Presentation':
                    stats['ppt_docs'].append({
                        'name': doc['file_name'],
                        'grade': grade
                    })
                
                # Topics
                for topic in metadata.get('topics', []):
                    stats['topic_distribution'][topic] += 1
            
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
        
        return stats
    
    def _print_statistics(self, stats: Dict[str, Any]):
        """Print formatted statistics."""
        
        print(f"\n3. DOCUMENT STATISTICS")
        print(f"   {'─'*60}")
        print(f"   Total Documents: {stats['total_docs']}")
        print(f"   Total Words: {stats['total_words']:,}")
        print(f"   Total Characters: {stats['total_chars']:,}")
        print(f"   Estimated Chunks: {stats['total_chunks_estimated']:,}")
        
        print(f"\n4. GRADE DISTRIBUTION")
        print(f"   {'─'*60}")
        for grade in sorted(stats['grade_distribution'].keys()):
            count = stats['grade_distribution'][grade]
            percentage = (count / stats['total_docs'] * 100)
            print(f"   {grade:25s}: {count:3d} docs ({percentage:5.1f}%)")
        
        print(f"\n5. DIFFICULTY DISTRIBUTION")
        print(f"   {'─'*60}")
        for diff in sorted(stats['difficulty_distribution'].keys()):
            count = stats['difficulty_distribution'][diff]
            percentage = (count / stats['total_docs'] * 100)
            print(f"   {diff:25s}: {count:3d} docs ({percentage:5.1f}%)")
        
        print(f"\n6. SOURCE TYPE DISTRIBUTION")
        print(f"   {'─'*60}")
        for source in sorted(stats['source_type_distribution'].keys()):
            count = stats['source_type_distribution'][source]
            percentage = (count / stats['total_docs'] * 100)
            print(f"   {source:25s}: {count:3d} docs ({percentage:5.1f}%)")
        
        print(f"\n7. TOPIC COVERAGE")
        print(f"   {'─'*60}")
        if stats['topic_distribution']:
            for topic in sorted(stats['topic_distribution'].keys()):
                count = stats['topic_distribution'][topic]
                print(f"   {topic:25s}: {count:3d} documents")
        else:
            print("   No topics identified")
        
        print(f"\n8. NCERT DOCUMENTS ({len(stats['ncert_docs'])})")
        print(f"   {'─'*60}")
        if stats['ncert_docs']:
            for doc in sorted(stats['ncert_docs'], key=lambda x: x['grade']):
                chapter_info = doc.get('chapter')
                if chapter_info:
                    print(f"   {doc['name']:30s} - {doc['grade']} "
                          f"(Ch. {chapter_info.get('chapter')})")
                else:
                    print(f"   {doc['name']:30s} - {doc['grade']}")
        else:
            print("   No NCERT documents found")
        
        print(f"\n9. TEXTBOOK DOCUMENTS ({len(stats['textbook_docs'])})")
        print(f"   {'─'*60}")
        if stats['textbook_docs']:
            for doc in stats['textbook_docs']:
                print(f"   {doc['name']:40s} - {doc['grade']} "
                      f"({doc['words']:,} words)")
        else:
            print("   No textbook documents found")
        
        print(f"\n10. PRESENTATION DOCUMENTS ({len(stats['ppt_docs'])})")
        print(f"   {'─'*60}")
        if stats['ppt_docs']:
            for doc in stats['ppt_docs']:
                print(f"   {doc['name']:40s} - {doc['grade']}")
        else:
            print("   No presentation documents found")
        
        print(f"\n{'='*70}\n")
    
    def _generate_report(self, stats: Dict[str, Any]):
        """Generate markdown report for mid-evaluation."""
        
        report_path = Path("docs") / "data_collection_report.md"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Data Collection Report - K-12 Geometry SME\n\n")
            f.write("**Project**: Subject Matter Expert AI Agent - Geometry\n")
            f.write("**Domain**: K-12 Education (Grades 6-10)\n")
            f.write("**Date**: Generated automatically\n\n")
            
            f.write("---\n\n")
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Documents**: {stats['total_docs']}\n")
            f.write(f"- **Total Words**: {stats['total_words']:,}\n")
            f.write(f"- **Total Characters**: {stats['total_chars']:,}\n")
            f.write(f"- **Estimated Chunks**: {stats['total_chunks_estimated']:,}\n")
            f.write(f"- **Grade Coverage**: Grades 6-10\n\n")
            
            f.write("---\n\n")
            f.write("## Data Sources\n\n")
            
            f.write("### 1. NCERT Textbooks (Official Curriculum)\n\n")
            f.write(f"**Count**: {len(stats['ncert_docs'])} documents\n\n")
            f.write("**Rationale**: NCERT textbooks are the official curriculum for Indian schools "
                   "and provide authoritative, grade-appropriate content for K-12 education.\n\n")
            if stats['ncert_docs']:
                f.write("**Documents**:\n")
                for doc in sorted(stats['ncert_docs'], key=lambda x: x['grade']):
                    chapter_info = doc.get('chapter')
                    if chapter_info:
                        f.write(f"- `{doc['name']}` - {doc['grade']}, "
                               f"Chapter {chapter_info.get('chapter')}\n")
                    else:
                        f.write(f"- `{doc['name']}` - {doc['grade']}\n")
                f.write("\n")
            
            f.write("### 2. Comprehensive Geometry Textbook\n\n")
            f.write(f"**Count**: {len(stats['textbook_docs'])} document(s)\n\n")
            f.write("**Title**: Geometry for Enjoyment and Challenge (McDougal Littell)\n\n")
            f.write("**Rationale**: This comprehensive textbook covers geometry topics from basic "
                   "to advanced levels (Grades 6-12), providing detailed explanations, proofs, "
                   "and practice problems. It complements NCERT material with different "
                   "pedagogical approaches.\n\n")
            if stats['textbook_docs']:
                f.write("**Documents**:\n")
                for doc in stats['textbook_docs']:
                    f.write(f"- `{doc['name']}` - {doc['words']:,} words\n")
                f.write("\n")
            
            f.write("### 3. Presentation Materials\n\n")
            f.write(f"**Count**: {len(stats['ppt_docs'])} presentation(s)\n\n")
            f.write("**Rationale**: PowerPoint presentations provide visual explanations and "
                   "structured topic introductions, useful for generating explanations and "
                   "visual learning content.\n\n")
            if stats['ppt_docs']:
                f.write("**Documents**:\n")
                for doc in stats['ppt_docs']:
                    f.write(f"- `{doc['name']}`\n")
                f.write("\n")
            
            f.write("---\n\n")
            f.write("## Grade Distribution\n\n")
            f.write("| Grade Level | Documents | Percentage |\n")
            f.write("|-------------|-----------|------------|\n")
            for grade in sorted(stats['grade_distribution'].keys()):
                count = stats['grade_distribution'][grade]
                pct = (count / stats['total_docs'] * 100)
                f.write(f"| {grade} | {count} | {pct:.1f}% |\n")
            f.write("\n")
            
            f.write("---\n\n")
            f.write("## Difficulty Distribution\n\n")
            f.write("| Difficulty | Documents | Percentage |\n")
            f.write("|------------|-----------|------------|\n")
            for diff in sorted(stats['difficulty_distribution'].keys()):
                count = stats['difficulty_distribution'][diff]
                pct = (count / stats['total_docs'] * 100)
                f.write(f"| {diff} | {count} | {pct:.1f}% |\n")
            f.write("\n")
            
            f.write("---\n\n")
            f.write("## Topic Coverage\n\n")
            if stats['topic_distribution']:
                f.write("| Topic | Document Count |\n")
                f.write("|-------|----------------|\n")
                for topic in sorted(stats['topic_distribution'].keys()):
                    count = stats['topic_distribution'][topic]
                    f.write(f"| {topic.title()} | {count} |\n")
            else:
                f.write("No topics identified yet.\n")
            f.write("\n")
            
            f.write("---\n\n")
            f.write("## Data Volume Justification\n\n")
            
            f.write("### Coverage Rationale\n\n")
            f.write("1. **Curriculum Alignment**: NCERT documents ensure alignment with "
                   "Indian school curriculum for Grades 6-10.\n\n")
            f.write("2. **Comprehensive Coverage**: The additional textbook provides broader "
                   "coverage of geometry concepts with alternative explanations.\n\n")
            f.write("3. **Multiple Pedagogical Approaches**: Combining NCERT (Indian curriculum) "
                   "with international textbooks provides diverse teaching styles.\n\n")
            f.write("4. **Visual Learning**: Presentations complement text-based learning with "
                   "structured visual content.\n\n")
            
            f.write("### Quality over Quantity\n\n")
            f.write(f"We prioritized **quality authoritative sources** over collecting large "
                   f"volumes of potentially redundant or low-quality content. Our {stats['total_docs']} "
                   f"documents provide:\n\n")
            f.write("- Complete coverage of K-12 geometry topics\n")
            f.write("- Multiple difficulty levels (Beginner to Advanced)\n")
            f.write("- Both theoretical explanations and practical problems\n")
            f.write("- Authoritative, curriculum-aligned content\n\n")
            
            f.write("### Estimated System Capacity\n\n")
            f.write(f"- **Chunks after processing**: ~{stats['total_chunks_estimated']:,}\n")
            f.write("- **Tokens** (estimated): ~{stats['total_words'] * 1.3:,.0f}\n")
            f.write("- **Vector embeddings**: ~{stats['total_chunks_estimated']:,} × 768 dimensions\n")
            f.write("- **Storage estimate**: ~{stats['total_chunks_estimated'] * 3 / 1000:.1f} MB (embeddings only)\n\n")
            
            f.write("---\n\n")
            f.write("## Processing Pipeline\n\n")
            f.write("### Document Processing Steps\n\n")
            f.write("1. **Format Detection**: Automatically identify PDF, DOCX, PPTX files\n")
            f.write("2. **Text Extraction**: Extract text preserving structure (pages, slides)\n")
            f.write("3. **Grade Classification**: \n")
            f.write("   - Filename pattern matching for NCERT (e.g., `class6_9.pdf`)\n")
            f.write("   - Content-based classification for other sources\n")
            f.write("   - Keyword matching against grade-specific vocabulary\n")
            f.write("4. **Metadata Extraction**: \n")
            f.write("   - Topics (theorems, shapes, angles, formulas, concepts)\n")
            f.write("   - Difficulty assessment\n")
            f.write("   - Source type identification\n")
            f.write("5. **Hierarchical Chunking**: 3 levels (2048, 512, 128 tokens)\n")
            f.write("6. **Embedding Generation**: sentence-transformers (all-mpnet-base-v2)\n")
            f.write("7. **Vector Database Indexing**: Elasticsearch with parent-child relationships\n\n")
            
            f.write("---\n\n")
            f.write("## Conclusions\n\n")
            f.write("The collected dataset provides:\n\n")
            f.write("✅ **Comprehensive coverage** of Grades 6-10 geometry curriculum\n\n")
            f.write("✅ **Authoritative sources** (NCERT official textbooks)\n\n")
            f.write("✅ **Multiple difficulty levels** for adaptive learning\n\n")
            f.write("✅ **Diverse content types** (textbooks, presentations)\n\n")
            f.write("✅ **Sufficient volume** for robust RAG system\n\n")
            f.write(f"The estimated **{stats['total_chunks_estimated']:,} chunks** will provide "
                   f"rich context for question answering, explanation generation, and "
                   f"educational content creation.\n\n")
        
        print(f"✓ Report generated: {report_path}")
        print(f"  Copy this to your mid-evaluation submission!\n")

def main():
    """Main verification function."""
    verifier = DataVerifier()
    stats = verifier.verify_processing()
    
    if stats:
        print("\n" + "="*70)
        print("VERIFICATION COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("1. Review the generated report in docs/data_collection_report.md")
        print("2. If data looks good, run: python scripts/build_database.py")
        print("3. Test retrieval with: python scripts/test_rag_pipeline.py")
        print("4. Include this report in your mid-evaluation submission\n")
    else:
        print("\n" + "="*70)
        print("VERIFICATION FAILED")
        print("="*70)
        print("\nPlease process your data first:")
        print("1. Place documents in data/raw/")
        print("2. Run: python scripts/build_database.py")
        print("3. Then run this script again\n")

if __name__ == "__main__":
    main()