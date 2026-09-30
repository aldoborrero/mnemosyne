"""Hermes CLI discovery adapter; command implementation lives in src/mnemosyne."""

import importlib.util
import sys
from pathlib import Path

# Hermes can load cli.py before __init__.py, with an absent or empty package.
# Load the real entry point so subsequent provider discovery also works.
_package = sys.modules.get(__package__)
if _package is None or not hasattr(_package, "register"):
    _plugin_dir = Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location(
        __package__,
        _plugin_dir / "__init__.py",
        submodule_search_locations=[str(_plugin_dir)],
    )
    if _spec is None or _spec.loader is None:
        raise ImportError("Cannot load the Mnemosyne plugin package")
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[__package__] = _module
    try:
        _spec.loader.exec_module(_module)
    except Exception:
        if _package is None:
            sys.modules.pop(__package__, None)
        else:
            sys.modules[__package__] = _package
        raise

from .commands import mnemosyne_command, register_cli

__all__ = ["mnemosyne_command", "register_cli"]
