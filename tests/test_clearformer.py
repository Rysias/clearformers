import pytest
import numpy as np
import pickle
from pathlib import Path
from bertopic import BERTopic
from explainlp.clearformer import Clearformer


def read_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def embeddings():
    """random embeddings in the range (-5, 5)"""
    return np.load(Path("tests/example_embeddings.npy"))


@pytest.fixture(scope="session")
def topic_model():
    return BERTopic.load(str(Path("tests/topic_model")))


@pytest.fixture(scope="session")
def clearformer(topic_model, embeddings):
    clearformer = Clearformer(topic_model)
    clearformer.fit(embeddings)
    return clearformer


def test_fit(clearformer, embeddings):
    clearformer.fit(embeddings)
    assert hasattr(clearformer, "centroids")
    assert clearformer.centroids.shape == (2, 5)


def test_clearformer_simple_transform(clearformer, embeddings):
    topx = clearformer.transform(embeddings)
    assert topx.shape == (500, 2)

