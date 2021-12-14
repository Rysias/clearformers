import pytest
import numpy as np
import pickle
from pathlib import Path
from bertopic import BERTopic
from explainlp.explainlp import ClearSearch
from explainlp.clearsify import Clearsifier
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
    return (np.random.rand(50, 757) - 0.5) * 10


@pytest.fixture(scope="session")
def probs():
    return np.random.rand(50)


@pytest.fixture(scope="session")
def topic_model():
    return BERTopic.load(str(Path("../TransTopicXAI/models/20211206_1223_topic_model")))


@pytest.fixture(scope="session")
def model(topic_model):
    return ClearSearch(topic_model=topic_model,)


@pytest.fixture(scope="session")
def clearformer(topic_model, embeddings):
    clearformer = Clearformer(topic_model)
    clearformer.fit(embeddings)
    return clearformer


def test_clearformer_train_transform(clearformer, embeddings, topics, probs):
    X = np.hstack((topics[:, np.newaxis], probs[:, np.newaxis], embeddings))
    topx = clearformer.transform(X)
    assert topx.shape == (50, 5)


def test_clearformer_simple_transform(clearformer, embeddings):
    topx = clearformer.transform(embeddings)
    assert topx.shape == (50, 5)

