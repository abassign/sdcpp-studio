# Vendored huggingface/gguf package.
#
# IMPORTANT (vendoring support): the internal modules of this package use
# absolute imports like `from gguf.constants import ...` (e.g. gguf_reader.py,
# metadata.py, vocab.py). To make those work regardless of how this package is
# imported (from `libs.gguf` or projected onto the top-level `gguf`), we add
# this directory to sys.path before importing the submodules. This keeps the
# vendored copy self-contained and relocatable: a fresh clone from GitHub works
# without installing anything and without a hardcoded /app path.
import os as _os
import sys as _sys

_GGUF_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _GGUF_DIR not in _sys.path:
    _sys.path.insert(0, _GGUF_DIR)

from .constants import *
from .lazy import *
from .gguf_reader import *
from .gguf_writer import *
from .quants import *
from .tensor_mapping import *
from .vocab import *
from .utility import *
from .metadata import *