"""Hermes directory entry point for MnemosyneMemoryProvider.

Implementation lives in src/mnemosyne; Hermes discovers this file by path.
Keep imports under the package name chosen by Hermes to isolate plugin copies.
"""

import sys
from pathlib import Path
from types import ModuleType

# Older Hermes loaders do not create the synthetic parent package.
_parent_name = __name__.rpartition(".")[0]
if _parent_name and _parent_name not in sys.modules:
    _parent = ModuleType(_parent_name)
    _parent.__path__ = []
    sys.modules[_parent_name] = _parent

__path__.insert(0, str(Path(__file__).resolve().parent / "src" / "mnemosyne"))

from .provider import MnemosyneMemoryProvider, register

__all__ = ["MnemosyneMemoryProvider", "register"]
