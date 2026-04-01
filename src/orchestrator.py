import logging
import numpy as np
import uuid
import time
from typing import Dict, Any, Tuple, List, Optional

# Import core modules (assuming they are in the same directory or Python path)
from src.preprocessing import FacePreprocessor
from src.vector_database import FaissVectorDB


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger("FaceRecSystem")


class Matcher:
    """
    Evaluates similarity scores returned by the Vector DB against strict thresholds.
    Handles secondary verification protocols like Metadata cross-checking.
    """
    def __init__(self, similarity_threshold: float = 0.45):
        # 0.45 - 0.50 is an industry-standard Cosine Similarity threshold for ArcFace (buffalo_l)
        self.threshold = similarity_threshold

    def evaluate_search_results(self, matched_ids: List[str], scores: List[float], target_user_id: str) -> Tuple[bool, float]:
        """Iterates through Top-K FAISS results to find the target user and verify the score."""
        for str_id, score in zip(matched_ids, scores):
            if str_id == target_user_id:
                is_match = score >= self.threshold
                logger.info(f"Similarity Score for {target_user_id}: {score:.4f} | Threshold: {self.threshold} | Match: {is_match}")
                return is_match, float(score)
                
        logger.warning(f"Target user {target_user_id} not found in top FAISS search results.")
        return False, 0.0

    def cross_check_metadata(self, input_meta: Dict[str, Any], db_meta: Dict[str, Any]) -> bool:
        """Cross-references biographical data (e.g., from an OCR'd ID scan vs User Input)."""
        logger.info("Performing Metadata Cross-check (DOB, Name)...")
        
        name_match = input_meta.get("name", "").strip().lower() == db_meta.get("name", "").strip().lower()
        dob_match = input_meta.get("dob", "").strip() == db_meta.get("dob", "").strip()
        
        if name_match and dob_match:
            logger.info("Metadata Cross-check: PASSED")
            return True
            
        logger.warning(f"Metadata Cross-check: FAILED. Input: {input_meta}, DB: {db_meta}")
        return False


class AuditLogger:
    """
    Logs transaction outcomes for compliance, fraud analysis, and system monitoring.
    In a production environment, this pushes data to Datadog, ELK, or a secure Postgres DB.
    """
    @staticmethod
    def log_transaction(tx_id: str, comparison_type: str, result: bool, details: Dict[str, Any]):
        logger.info("\n" + "="*50)
        logger.info(f"AUDIT LOG | TX_ID: {tx_id}")
        logger.info(f"TYPE: {comparison_type} | VERIFIED: {result}")
        for k, v in details.items():
            logger.info(f"  -> {k}: {v}")
        logger.info("="*50 + "\n")


class FaceRecognitionSystem:
    """
    The main controller tying together the Dual Comparison Flow (Live<->Live, Live<->Doc).
    Exposes clean endpoints for enrollment and verification.
    """
    def __init__(self, similarity_threshold: float = 0.45):
        logger.info("Initializing Face Recognition System Orchestrator...")
        
        # 1. Initialize ML Pipeline
        self.preprocessor = FacePreprocessor()
        
        # 2. Initialize Databases
        self.live_db = FaissVectorDB("Live_Gallery_DB", embedding_dim=512)
        self.doc_db = FaissVectorDB("Document_ID_DB", embedding_dim=512)
        
        # 3. Initialize Verification Logic
        self.matcher = Matcher(similarity_threshold=similarity_threshold) 

    def enroll_document(self, user_id: str, doc_image: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Processes an ID document and stores its embedding and metadata."""
        processed = self.preprocessor.process(doc_image, source="DOC")
        if processed and "embedding" in processed:
            self.doc_db.insert(user_id, processed["embedding"], metadata)
            return True
        logger.error(f"Failed to enroll document for user: {user_id}")
        return False
            
    def enroll_live(self, user_id: str, live_image: np.ndarray) -> bool:
        """Processes a live baseline face and stores its embedding in the gallery."""
        processed = self.preprocessor.process(live_image, source="LIVE")
        if processed and "embedding" in processed:
            self.live_db.insert(user_id, processed["embedding"])
            return True
        logger.error(f"Failed to enroll live reference for user: {user_id}")
        return False

    def verify(self, input_image: np.ndarray, target_user_id: str, mode: str = "LIVE_TO_LIVE", input_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Executes the verification pipeline.
        Args:
            input_image: The BGR numpy array from the camera.
            target_user_id: The ID claimed by the user.
            mode: "LIVE_TO_LIVE" (Authentication) or "LIVE_TO_DOC" (KYC/Onboarding).
            input_metadata: Required if mode is "LIVE_TO_DOC" for secondary verification.
        """
        start_time = time.time()
        tx_id = str(uuid.uuid4())[:8]
        logger.info(f"--- Starting Verification Request [{tx_id}] | Mode: {mode} ---")
        
        # 1. Preprocessing & Liveness Detection
        processed = self.preprocessor.process(input_image, source="LIVE")
        if not processed:
            AuditLogger.log_transaction(tx_id, mode, False, {"error": "Preprocessing, Detection, or Liveness failed", "processing_time_ms": int((time.time() - start_time)*1000)})
            return False
            
        probe_embedding = processed["embedding"]
        is_verified = False
        sim_score = 0.0
        details = {}
        
        # 2. Vector DB Search & Matching
        if mode == "LIVE_TO_LIVE":
            # Fast FAISS Search in the Live Gallery
            matched_ids, scores = self.live_db.search(probe_embedding, top_k=5)
            is_verified, sim_score = self.matcher.evaluate_search_results(matched_ids, scores, target_user_id)
            if not is_verified and sim_score == 0.0:
                details["error"] = "User not found or similarity too low in Live DB"
                
        elif mode == "LIVE_TO_DOC":
            # Fast FAISS Search in the Document DB
            matched_ids, scores = self.doc_db.search(probe_embedding, top_k=5)
            face_match, sim_score = self.matcher.evaluate_search_results(matched_ids, scores, target_user_id)
            
            # Cross-check Metadata
            target_metadata = self.doc_db.get_metadata(target_user_id)
            meta_match = False
            
            if target_metadata and input_metadata:
                meta_match = self.matcher.cross_check_metadata(input_metadata, target_metadata)
            else:
                details["warning"] = "Metadata missing for cross-check"
            
            is_verified = face_match and meta_match
            details["meta_match"] = meta_match
            details["face_match"] = face_match
        else:
            details["error"] = f"Unknown verification mode: {mode}"

        # 3. Post-processing Logging
        details["similarity_score"] = float(sim_score)
        details["processing_time_ms"] = int((time.time() - start_time) * 1000)
        
        AuditLogger.log_transaction(tx_id, mode, is_verified, details)
        
        return is_verified



if __name__ == "__main__":
    import cv2
    logger.info("Running orchestrator sanity check. For full tests, use tests/visualize_result.py")
    
    # Create dummy image data to ensure the class initializes correctly
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    try:
        system = FaceRecognitionSystem()
        logger.info("System initialized successfully! Ready for production routing.")
    except Exception as e:
        logger.error(f"Failed to initialize FaceRecognitionSystem: {e}")