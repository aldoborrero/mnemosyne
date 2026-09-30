"""Composite provider tool definitions, exposure defaults and routing metadata.

Execution and write-policy enforcement belong to provider.py. The approved-writes
provider keeps its separate tool surface in approved.py.
"""

from .forget import MEMORY_FORGET_SCHEMA

__all__ = [
    "DEFAULT_TOOL_NAMES",
    "RECALL_QUERY_MAX_CHARS",
    "TOOL_DISPATCH",
    "TOOL_SCHEMAS",
    "TOOL_TIMEOUT_KEYS",
    "WRITE_TOOLS",
]


# ---------------------------------------------------------------------------
# Curated tool schemas (plan item 6 — tightened role descriptions)
# ---------------------------------------------------------------------------

_PROFILE_SCHEMA = {
    "name": "memory_profile",
    "description": (
        "ONLY for the user's profile card: name, role, communication style, "
        "stable preferences. Read or update. Do NOT use for general facts or "
        "past-conversation history — for those use memory_recall."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "card": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New card as a list of fact strings. Omit to read.",
            },
        },
        "required": [],
    },
}

_REASONING_SCHEMA = {
    "name": "memory_reasoning",
    "description": (
        "ONLY questions about the user as a person: their style, habits, "
        "behavioral patterns, what approach works best with them. NOT for "
        "general knowledge and NOT for facts from past conversations — for "
        "those use memory_recall or memory_reflect."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language question about the user as a person.",
            },
            "reasoning_level": {
                "type": "string",
                "enum": ["minimal", "low", "medium", "high", "max"],
                "description": "Depth control. Omit for default (low).",
            },
        },
        "required": ["query"],
    },
}

_CONCLUDE_SCHEMA = {
    "name": "memory_conclude",
    "description": (
        "Record a stable user-related conclusion (preference, habit, style). "
        "NOT for technical facts or events — those go through memory_recall."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {
                "type": "string",
                "description": "The conclusion to persist.",
            },
        },
        "required": ["conclusion"],
    },
}

_RECALL_SCHEMA = {
    "name": "memory_recall",
    "description": (
        "FIRST CHOICE for 'do you remember when we did X?', 'we discussed this', "
        "'how did we fix that before'. Multi-strategy search (semantic + entity "
        "graph) over all past conversations. Returns relevant facts and "
        "fragments. THIS IS THE MAIN LONG-TERM MEMORY TOOL."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for."},
            "max_tokens": {
                "type": "integer",
                "description": "Token budget (default 800, max 4096).",
            },
        },
        "required": ["query"],
    },
}

_REFLECT_SCHEMA = {
    "name": "memory_reflect",
    "description": (
        "LLM synthesis across past-conversation facts. Use when you need a "
        "summary spanning multiple sources ('what did we conclude about X?', "
        "'what facts do we have on topic Y?'). NOT for questions about the "
        "user as a person."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language question."},
        },
        "required": ["query"],
    },
}


# Maps curated tool names → (inner_provider_attr, inner_tool_name).
TOOL_DISPATCH = {
    "memory_profile": ("honcho", "honcho_profile"),
    "memory_reasoning": ("honcho", "honcho_reasoning"),
    "memory_conclude": ("honcho", "honcho_conclude"),
    "memory_recall": ("hindsight", "hindsight_recall"),
    "memory_reflect": ("hindsight", "hindsight_reflect"),
}


TOOL_SCHEMAS = {
    "memory_profile": _PROFILE_SCHEMA,
    "memory_reasoning": _REASONING_SCHEMA,
    "memory_conclude": _CONCLUDE_SCHEMA,
    "memory_recall": _RECALL_SCHEMA,
    "memory_reflect": _REFLECT_SCHEMA,
    "memory_forget": MEMORY_FORGET_SCHEMA,
}

# Catalogue insertion order is the default order exposed to Hermes.
DEFAULT_TOOL_NAMES = tuple(TOOL_SCHEMAS)

# Tools that write to a backend directly rather than through the bridge.
WRITE_TOOLS = ("memory_conclude", "memory_forget")


# Tool name → configuration key within the timeouts section.
TOOL_TIMEOUT_KEYS = {
    "memory_recall": "recall",
    "memory_reasoning": "reasoning",
    "memory_reflect": "reflect",
    "memory_profile": "profile",
    "memory_conclude": "conclude",
    "memory_forget": "forget",
}


# Hindsight enforces "Query too long: N tokens exceeds maximum of 500".
# 1500 chars ≈ 350-450 tokens (RU runs higher chars/token than EN); leaves
# headroom for any query expansion Hindsight does internally.
RECALL_QUERY_MAX_CHARS = 1500
