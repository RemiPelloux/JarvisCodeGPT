from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

class VectorStore(ABC):
    """A base class for vector stores"""

    @abstractmethod
    def __init__(self, docs: List[Any]):
        pass

    @abstractmethod
    def get_nns_by_vector(
        self, vector: List[float], n: int, include_distances: bool = False
    ) -> List[int]:
        pass

    def get_nns_by_id(
        self, doc_id: int, n: int, include_distances: bool = False
    ) -> List[Tuple[int, float]]:
        vector = self.get_vector(doc_id)
        results = self.get_nns_by_vector(vector, n + 1, include_distances)
        return [
            (result, distance)
            for result, distance in results
            if result != doc_id
        ][:n]

    @abstractmethod
    def get_vector(self, doc_id: int) -> List[float]:
        pass

    @abstractmethod
    def get_all_documents(self) -> List[str]:
        pass

    def get_vectors(self, doc_ids: List[int]) -> Dict[int, List[float]]:
        return {doc_id: self.get_vector(doc_id) for doc_id in doc_ids}

    def build_index(self, index_params: Dict[str, Any]):
        pass
