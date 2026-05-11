"""Public package initialization."""

import logging

from afmlevel._version import __version__ as __version__

# Prevent "No handler found" in user apps and keep library quiet by default.
logging.getLogger(__name__).addHandler(logging.NullHandler())
