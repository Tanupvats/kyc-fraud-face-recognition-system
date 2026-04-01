import logging

try:
    from insightface.app import FaceAnalysis
except ImportError:
    raise ImportError("Please install insightface: pip install insightface onnxruntime")

logger = logging.getLogger("FaceRecSystem")

class MLEngine:
    """
    Singleton-like wrapper for the InsightFace model to prevent reloading.
    Handles the heavy lifting for Face Detection (RetinaFace) and Embedding Extraction (ArcFace).
    """
    _instance = None
    _app = None

    @classmethod
    def get_app(cls):
        if cls._app is None:
            logger.info("Loading InsightFace 'buffalo_l' model (RetinaFace + ArcFace)...")
            # buffalo_l includes detection (retinaface) and recognition (arcface)
            cls._app = FaceAnalysis(name='buffalo_l', root='~/.insightface', providers=['CPUExecutionProvider'])
            # det_size=(640,640) is standard for generic face detection
            cls._app.prepare(ctx_id=0, det_size=(640, 640)) 
            logger.info("ML Engine Loaded Successfully.")
        return cls._app