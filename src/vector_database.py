import logging
import numpy as np
import faiss
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger("FaceRecSystem")

class FaissVectorDB:
    """
    Production-grade Vector DB using FAISS.
    Handles 'Live Image DB' and 'Doc Image DB'.
    """
    def __init__(self, db_name: str, embedding_dim: int = 512):
        self.db_name = db_name
        self.embedding_dim = embedding_dim
        
        # Inner Product index (since ArcFace embeddings are L2 normalized, IP == Cosine Similarity)
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
        
        # FAISS requires integer IDs. We maintain a mapping between string IDs and int IDs.
        self._str_to_int: Dict[str, int] = {}
        self._int_to_str: Dict[int, str] = {}
        self._current_int_id = 0
        
        self.metadata: Dict[str, Dict[str, Any]] = {}
        logger.info(f"Initialized FAISS Vector DB: {self.db_name} (Dim: {self.embedding_dim})")

    def _get_or_create_int_id(self, record_id: str) -> int:
        if record_id not in self._str_to_int:
            self._current_int_id += 1
            self._str_to_int[record_id] = self._current_int_id
            self._int_to_str[self._current_int_id] = record_id
        return self._str_to_int[record_id]

    def insert(self, record_id: str, vector: np.ndarray, meta: Optional[Dict] = None):
        int_id = self._get_or_create_int_id(record_id)
        
        # Ensure vector shape is (1, dim) and type is float32 for FAISS
        vec_np = np.array([vector], dtype=np.float32)
        
        # Add to FAISS index
        self.index.add_with_ids(vec_np, np.array([int_id], dtype=np.int64))
        
        if meta:
            self.metadata[record_id] = meta
        logger.info(f"[{self.db_name}] Inserted record: {record_id}")

    def search(self, query_vector: np.ndarray, top_k: int = 1) -> Tuple[List[str], List[float]]:
        """Returns lists of matched string IDs and their cosine similarity scores."""
        if self.index.ntotal == 0:
            return [], []
            
        vec_np = np.array([query_vector], dtype=np.float32)
        distances, int_ids = self.index.search(vec_np, top_k)
        
        str_ids = [self._int_to_str[i] for i in int_ids[0] if i != -1]
        scores = [dist for dist in distances[0] if dist != -1.0]
        
        return str_ids, scores

    def get_metadata(self, record_id: str) -> Optional[Dict]:
        return self.metadata.get(record_id)