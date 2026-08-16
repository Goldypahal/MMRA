"""
tasks_extended.py — Companion to repo root for MMRA extended adversarial test set.
Re-exports EXTENDED_TASKS and helper functions from src.tasks_extended.
"""

from src.tasks_extended import (
    EXTENDED_TASKS,
    MATH_EXT,
    LOGIC_EXT,
    CODING_EXT,
    SCIENCE_EXT,
    LANGUAGE_EXT,
    KNOWLEDGE_EXT,
    OPENENDED_EXT,
    get_extended_by_category,
    get_extended_by_difficulty,
)

__all__ = [
    "EXTENDED_TASKS",
    "MATH_EXT",
    "LOGIC_EXT",
    "CODING_EXT",
    "SCIENCE_EXT",
    "LANGUAGE_EXT",
    "KNOWLEDGE_EXT",
    "OPENENDED_EXT",
    "get_extended_by_category",
    "get_extended_by_difficulty",
]
