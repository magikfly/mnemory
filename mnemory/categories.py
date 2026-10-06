"""Predefined memory categories and category management.

Categories follow a hybrid model:
- Predefined top-level categories (the LLM knows these exist)
- Dynamic subcategories via prefix namespacing (e.g., project:<name>)

The predefined set ensures LLMs can discover and filter categories reliably.
The project: namespace allows dynamic, project-scoped memories without
requiring configuration changes.
"""

from __future__ import annotations

import logging

from mnemory.sanitize import validate_category_name

logger = logging.getLogger("mnemory.categories")

# Predefined categories with descriptions.
# These are always available and returned by list_categories even if empty.
PREDEFINED_CATEGORIES: dict[str, str] = {
    "personal": "Personal life, family, relationships, life events",
    "preferences": "Likes, dislikes, style, communication preferences",
    "health": "Physical, mental, medical, fitness, diet",
    "work": "Job, career, professional, colleagues",
    "technical": "Tools, languages, infrastructure, patterns, debugging",
    "finance": "Money, investments, billing, expenses",
    "home": "House, apartment, appliances, maintenance, repairs",
    "vehicles": "Cars, bikes, maintenance history, insurance",
    "travel": "Trips, places, itineraries, transportation",
    "entertainment": "Movies, music, books, games, hobbies",
    "goals": "Objectives, plans, ambitions, deadlines",
    "decisions": "Conclusions, choices made, reasoning, trade-offs",
    "project": "Project-specific memories (use project:<name> format)",
}

# Valid memory types
VALID_MEMORY_TYPES = ("preference", "fact", "episodic", "procedural", "context")

# Valid importance levels with numeric weights for search reranking
IMPORTANCE_WEIGHTS: dict[str, float] = {
    "low": 0.1,
    "normal": 0.4,
    "high": 0.7,
    "critical": 1.0,
}


def importance_levels_at_or_above(min_level: str) -> list[str]:
    """Return importance level names at or above the given minimum.

    Example: importance_levels_at_or_above("high") -> ["high", "critical"]
    """
    min_weight = IMPORTANCE_WEIGHTS.get(min_level, 0.4)
    return [
        level for level, weight in IMPORTANCE_WEIGHTS.items() if weight >= min_weight
    ]


def validate_categories(categories: list[str]) -> list[str]:
    """Validate and normalize a list of categories.

    Rules:
    - Predefined categories are accepted as-is (lowercase)
    - project:<name> format is accepted for any <name>
    - Unknown top-level categories are rejected

    Returns the validated list (lowercased).
    Raises ValueError for invalid categories.
    """
    validated = []
    for cat in categories:
        cat = cat.strip().lower()
        if not cat:
            continue

        # Check for namespaced category (e.g., project:domecek/k8s-manifests)
        if ":" in cat:
            prefix, name = cat.split(":", 1)
            if prefix not in PREDEFINED_CATEGORIES:
                raise ValueError(
                    f"Unknown category prefix '{prefix}' in '{cat}'. "
                    f"Valid prefixes: {', '.join(sorted(PREDEFINED_CATEGORIES))}"
                )
            # Validate the name portion to prevent prompt injection
            # via category names that get interpolated into LLM prompts
            validate_category_name(name)
            validated.append(cat)
        elif cat in PREDEFINED_CATEGORIES:
            validated.append(cat)
        else:
            raise ValueError(
                f"Unknown category '{cat}'. "
                f"Valid categories: {', '.join(sorted(PREDEFINED_CATEGORIES))}"
            )

    return validated


def sanitize_categories(categories: list[str] | None) -> list[str]:
    """Coerce LLM-produced categories into the valid taxonomy.

    LOCAL-PATCH: model output is not a validated API input. The extractor
    invents tokens outside the taxonomy (observed: 'fact'), and the strict
    validate_categories() raise in add_memory() aborts an ENTIRE consolidation
    batch over one bad token. Drop unknown entries with a warning and keep the
    rest. User-supplied API categories keep going through validate_categories()
    unchanged, so API strictness is untouched.
    """
    if not categories:
        return []

    valid: list[str] = []
    for cat in categories:
        try:
            valid.extend(validate_categories([cat]))
        except ValueError:
            logger.warning(
                "Dropping invalid category %r (valid: %s)",
                cat,
                ", ".join(sorted(PREDEFINED_CATEGORIES)),
            )
    return valid


def validate_memory_type(memory_type: str) -> str:
    """Validate a memory type string."""
    memory_type = memory_type.strip().lower()
    if memory_type not in VALID_MEMORY_TYPES:
        raise ValueError(
            f"Unknown memory_type '{memory_type}'. "
            f"Valid types: {', '.join(VALID_MEMORY_TYPES)}"
        )
    return memory_type


def validate_importance(importance: str) -> str:
    """Validate an importance level string."""
    importance = importance.strip().lower()
    if importance not in IMPORTANCE_WEIGHTS:
        raise ValueError(
            f"Unknown importance '{importance}'. "
            f"Valid levels: {', '.join(IMPORTANCE_WEIGHTS)}"
        )
    return importance


def matches_category_filter(
    memory_categories: list[str], filter_categories: list[str]
) -> bool:
    """Check if a memory's categories match a filter.

    Matching rules:
    - "project" matches all "project:*" entries (prefix match)
    - "project:foo" matches exactly "project:foo"
    - "personal" matches exactly "personal"
    - Multiple filter categories use OR logic (match any)
    """
    if not filter_categories:
        return True

    for fc in filter_categories:
        fc = fc.lower()
        for mc in memory_categories:
            mc = mc.lower()
            if fc == mc:
                return True
            # Prefix match: filter "project" matches memory "project:foo"
            if ":" not in fc and mc.startswith(fc + ":"):
                return True
    return False


def count_categories(memories: list[dict]) -> dict[str, int]:
    """Count memories per category from a list of memory results.

    Returns a dict mapping category name to count. Includes both
    predefined categories and any dynamic subcategories found.
    """
    counts: dict[str, int] = {}
    for mem in memories:
        metadata = mem.get("metadata") or {}
        cats = metadata.get("categories", [])
        if isinstance(cats, list):
            for cat in cats:
                cat = cat.lower()
                counts[cat] = counts.get(cat, 0) + 1
    return counts

