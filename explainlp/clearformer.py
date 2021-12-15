import numpy as np
from bertopic import BERTopic
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union, Sequence, Iterable, Dict, List


class Clearformer(BaseEstimator, TransformerMixin):
    def __init__(self, topic_model: BERTopic) -> None:
        self.topic_model = topic_model
        self.nr_topics = self.topic_model.nr_topics

    def fit(self, X: np.ndarray, y=None):
        """Fits the centroids to the data"""
        # How to solve this weird overhead of double UMAP?
        umap_embeddings = self.topic_model.umap_model.transform(X)
        docs = ["i" for _ in range(X.shape[0])]
        topics, probs = self.topic_model.transform(docs, X)
        self.centroids = np.zeros(
            (self.nr_topics, umap_embeddings.shape[1])
        )  # Centroids need dimensions (number of topics, embedding-dimensionality)
        for i in range(self.nr_topics):
            self.centroids[i, :] += self._find_centroid(
                umap_embeddings, topics, probs, i
            )
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        featurizes multiple documents where each document is a numpy array of shape (n_paragraphs, embedding_dim)
        Also works if input is a (n_docs, embedding_dim) numpy array
        """
        try:
            umap_embeddings = self.topic_model.umap_model.transform(X)
            return cosine_similarity(umap_embeddings, self.centroids)
        except ValueError:
            raise

    def weighted_mean(
        self, X: np.ndarray, weights: Sequence[Union[int, float]]
    ) -> np.ndarray:
        return np.dot(X.T, weights) / np.sum(weights)

    def _find_centroid(
        self,
        embeddings: np.ndarray,
        topics: np.ndarray,
        probs: np.ndarray,
        target_topic: int,
    ) -> np.ndarray:
        """
        Arguments:
            embeddings: 2d with dimensions (num_documents, num_dimensions)
            topics: list of length num documents
            probs: np.array of length num_documents showing the probability of the assigned topic
            target_topic: the topic, we want to find the centroid for
        returns:
            The centroid for the cluster
        """
        if probs is None:  # Simple average
            return np.mean(embeddings[topics == target_topic, :], axis=0)
        # Filtering the embeddings
        filtered_embeddings = embeddings[topics == target_topic, :]
        filtered_probs = probs[topics == target_topic]

        # Calculating the centroid
        return self.weighted_mean(filtered_embeddings, filtered_probs)
