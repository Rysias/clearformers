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
def topics():
    return np.random.choice(list(range(-1, 10)), 50)


@pytest.fixture(scope="session")
def embeddings():
    """random embeddings in the range (-5, 5)"""
    return (np.random.rand(50, 384) - 0.5) * 10


@pytest.fixture(scope="session")
def probs():
    return np.random.rand(50)


@pytest.fixture(scope="session")
def topic_model():
    return BERTopic.load(str(Path("tests/topic_model")))


@pytest.fixture(scope="session")
def clearformer(topic_model, embeddings):
    clearformer = Clearformer(topic_model)
    clearformer.fit(embeddings)
    return clearformer


def test_clearformer_simple_transform(clearformer, embeddings):
    topx = clearformer.transform(embeddings)
    assert topx.shape == (50, 2)

