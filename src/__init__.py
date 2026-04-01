

import logging

# Initialize module-level logger
logger = logging.getLogger("FaceRecSystem")
logger.addHandler(logging.NullHandler()) # Prevent logging errors if not configured by the parent app

# Expose core components at the package level for clean importing
try:
    from .orchestrator import FaceRecognitionSystem
    from .preprocessing import FacePreprocessor
    from .extraction import MLEngine
    from .vector_database import FaissVectorDB

    __all__ = [
        "FaceRecognitionSystem",
        "FacePreprocessor",
        "MLEngine",
        "FaissVectorDB"
    ]
except ImportError as e:
    logger.warning(f"Failed to import core modules in src/__init__.py. Ensure dependencies are installed. Error: {e}")