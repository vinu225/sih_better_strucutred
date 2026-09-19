"""
SatQuery: Earth Observation Multimodal VLM & AI Agent Framework.
"""
import sys

__version__ = "1.0.0"

# Provide alias so legacy imports of 'satquery' resolve to 'backend'
sys.modules.setdefault("satquery", sys.modules[__name__])

