import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple

from src.extraction import MLEngine

logger = logging.getLogger("FaceRecSystem")

class FacePreprocessor:
    """
    Handles Detection, Quality Checks, and Liveness.
    Note: InsightFace handles internal cropping and alignment implicitly during extraction.
    """
    def __init__(self, min_det_score: float = 0.6):
        self.app = MLEngine.get_app()
        self.min_det_score = min_det_score
        logger.info("Face Preprocessor initialized.")

    def process(self, image: np.ndarray, source: str = "LIVE") -> Optional[Dict[str, Any]]:
        logger.info(f"Preprocessing image from source: {source}")
        
        # 1. Detect, Align, and Extract (InsightFace handles this internally in one pass)
        faces = self.app.get(image)
        
        if len(faces) == 0:
            logger.warning("No face detected.")
            return None
            
        # If multiple faces, pick the largest one (assuming the primary subject)
        if len(faces) > 1:
            logger.warning(f"Multiple faces ({len(faces)}) detected. Selecting the largest one.")
            faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)
            
        primary_face = faces[0]

        # 2. Quality / Confidence Check
        if primary_face.det_score < self.min_det_score:
            logger.warning(f"Face detection confidence ({primary_face.det_score:.2f}) below threshold ({self.min_det_score}).")
            return None
            
        # 3. Liveness Check (Only applicable for LIVE camera input)
        if source == "LIVE":
            is_live, liveness_score = self._liveness_check(image, primary_face.bbox)
            if not is_live:
                logger.warning(f"Liveness check failed! Score: {liveness_score:.4f}")
                return None
        
        return {
            "status": "success",
            "bbox": primary_face.bbox,
            "landmarks": primary_face.landmark_2d_106,
            "embedding": primary_face.normed_embedding, # Pre-normalized 512D ArcFace vector
            "liveness_passed": True if source == "LIVE" else "N/A"
        }

    def _liveness_check(self, full_image: np.ndarray, bbox: np.ndarray) -> Tuple[bool, float]:
        """
        PRODUCTION NOTE: 
        Integrate a dedicated Anti-Spoofing model here (e.g., MiniFASNet via ONNX).
        Example flow: crop bbox -> pass to ONNX model -> return real/fake score.
        For this script, we simulate a passing score to allow the pipeline to run.
        """
        # TODO: Load and run Silent-Face-Anti-Spoofing ONNX model
        mock_score = 0.95 
        return mock_score > 0.80, mock_score