from typing import List

import numpy as np
import faiss

from langchain.docstore.document import Document
from langchain.vectorstores.vectorstore import VectorStore

class FaissVectorStore(VectorStore):
    """A vector store implementation based on FAISS"""

    def __init__(self, embedding_size: int, metric_type: int = faiss.METRIC_INNER_PRODUCT):
        self.index = None
        self.doc_ids = []
        self.embedding_size = embedding_size
        self.metric_type = metric_type

    def add_vectors(self, vectors: np.ndarray) -> None:
        """Add a list of vectors to the index"""
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_size)
        ids = np.arange(len(self.doc_ids), len(self.doc_ids) + len(vectors))
        self.index.add(vectors)
        self.doc_ids.extend(ids)

    def search(self, query: np.ndarray, k: int) -> List[Document]:
        """Search for the k nearest neighbors of a query vector"""
        distances, indices = self.index.search(query, k)
        return [Document(doc_id=self.doc_ids[idx], score=distances[i]) for i, idx in enumerate(indices)]
