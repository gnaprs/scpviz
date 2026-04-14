"""Log / text formatting helpers for scpviz utilities."""
from __future__ import annotations


def format_log_prefix(level: str, indent: int | None = None) -> str:
    """
    Return a standardized log prefix with emoji and label.

    This helper formats log message prefixes consistently across scpviz,
    with optional indentation for nested output. Used internally for
    user-facing messages, warnings, errors, and updates.

    Args:
        level (str): Logging level keyword. Must be one of:

            - `"user"`: 🧭 [USER]  
            - `"search"`: 🔍 [SEARCH]  
            - `"info"`: ℹ️ [INFO]  
            - `"result"`: ✅ [OK]  
            - `"warn"`: ⚠️ [WARN]  
            - `"error"`: ❌ [ERROR]  
            - `"info_only"`: ℹ️  
            - `"filter_conditions"`: 🔸 (indented)  
            - `"result_only"`: ✅  
            - `"blank"`: empty string  
            - `"update"`: 🔄 [UPDATE]  
            - `"api"`: 🌐 [API]
            - `"update_only"`: 🔄  
            - `"warn_only"`: ⚠️
            - `"user_only"`: 🧭

        indent (int or None, optional): Indentation level override. Options:

            - `1`: no indent  
            - `2`: 5 spaces  
            - `3`: 10 spaces  

            If None, uses built-in default spacing (applied to most levels).

    Returns:
        str (str): A formatted log prefix with emoji and label.

    Raises:
        ValueError: If an unknown `level` string is provided.

    Example:
        Format an info prefix with default spacing:
        ```python
        from scpviz.utils import format_log_prefix
        format_log_prefix("info")
        ```

        ```
            ℹ️ [INFO]
        ```

        Format a warning prefix with explicit indent:
        ```python
        format_log_prefix("warn", indent=3)
        ```

        ```
                ⚠️ [WARN]
        ```
    """
    level = level.lower()
    base_prefixes = {
        "user": "🧭 [USER]",
        "search": "🔍 [SEARCH]",
        "info": "ℹ️ [INFO]",
        "result": "✅ [OK]",
        "warn": "⚠️ [WARN]",
        "error": "❌ [ERROR]",
        "info_only": "ℹ️",
        "filter_conditions": "     🔸 ",
        "result_only": "✅",
        "blank": "",
        "update": "🔄 [UPDATE]",
        "api": "🌐 [API]",
        "update_only": "🔄",
        "warn_only": "⚠️",
        "user_only": "🧭"
    }

    if level not in base_prefixes:
        raise ValueError(f"Unknown log level: {level}")

    prefix = base_prefixes[level]

    if indent is None:
        # Use default built-in spacing for all except info_only
        if level in ["info", "search", "result", "warn", "error"]:
            return "     " + prefix
        else:
            return prefix  # Default case, no indent (e.g. info_only)
    else:
        # Explicit indent override
        indent_spaces = {1: 0, 2: 5, 3: 10}
        space = " " * indent_spaces.get(indent, 0)
        return f"{space}{prefix}"
