# src/__init__.py (UPDATED)

"""
Geometry Subject Matter Expert (SME) System
K-12 Education - Geometry (Shapes, Angles, Theorems)

Phase 2: RAG Pipeline & Retrieval
"""

from . import config
from . import data_preparation
from . import database
from . import retrieval

__all__ = [
    'config',
    'data_preparation',
    'database',
    'retrieval'
]

__version__ = '0.2.0'
__author__ = 'Your Team Name'
__description__ = 'K-12 Geometry SME with RAG capabilities'
