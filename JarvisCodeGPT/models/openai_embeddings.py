from typing import Any, Iterable, List, Tuple

from sklearn.metrics.pairwise import cosine_similarity
import openai
from langchain.docstore.document import Document
from langchain.vectorstores import VectorStore
from openai.error import OpenAIError
import numpy as np


class OpenAIEmbeddings:
    """A wrapper for the OpenAI API to generate embeddings"""

    def __init__(self, openai_api_key: str, max_batch_size: int = 16):
        openai.api_key = openai_api_key
        self.model_engine = "text-davinci-002"
        self.max_batch_size = max_batch_size

    def __call__(self, text: str | List[str], **kwargs: Any) -> List[Tuple[int, List[float]]]:
        """Generate embeddings for a list of strings"""
        if isinstance(text, str):
            text = [text]

        prompt = "\n".join(text)
        prompt += "\n"
        prompt += f'Model: {self.model_engine}'

        try:
            response = openai.Completion.create(
                engine=self.model_engine,
                prompt=prompt,
                max_tokens=512,
                n=1,
                stop=None,
                temperature=0.5,
            )

            return [
                (i, document.embedding)
                for i, document in enumerate(
                    VectorStore([Document(page_content=choice.text)])
                )
                for choice in response.choices
            ]

        except OpenAIError as e:
            raise OpenAIError("Erreur lors de l'appel de l'API OpenAI. Veuillez vérifier votre clé API.") from e

    def embed(self, text: str) -> np.ndarray:
        """Embeds a string of text using OpenAI API"""
        response = self(text)
        embedding = np.array(response[0][1])
        return embedding

    def bulk_embed(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of strings using OpenAI API"""
        embeddings = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            response = self(batch)
            batch_embeddings = [np.array(r[1]) for r in response]
            embeddings += batch_embeddings
        embeddings = np.array(embeddings)
        return embeddings

    def similarity(self, text1: str, text2: str) -> float:
        """Computes cosine similarity between two strings"""
        emb1, emb2 = self.embed(text1), self.embed(text2)
        similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1)).item()
        return similarity

    def bulk_similarity(self, queries: List[str], docs: Iterable[Document]) -> List[List[float]]:
        """Computes cosine similarity between a list of queries and a list of documents"""
        query_embeddings = self.bulk_embed(queries)
        doc_embeddings = np.array([doc.embedding for doc in docs])
        similarities = cosine_similarity(query_embeddings, doc_embeddings)
        similarities = similarities.tolist()
        return similarities
