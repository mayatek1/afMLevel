"""Public package initialization."""

import logging

# Prevent "No handler found" in user apps and keep library quiet by default.
logging.getLogger(__name__).addHandler(logging.NullHandler())
