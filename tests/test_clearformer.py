import pytest
import numpy as np
import pickle
from pathlib import Path
from bertopic import BERTopic
from explainlp.clearformer import Clearformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


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


def test_pipeline(clearformer, embeddings):
    pipe = make_pipeline(clearformer, MinMaxScaler())
    embs = pipe.fit_transform(embeddings)
    assert np.max(embs == 1)
    assert embs.shape == (500, 2)


def test_pipeline(clearformer, embeddings):
    y = np.random.choice([1, 0], size=embeddings.shape[0])
    pipe = make_pipeline(clearformer, MinMaxScaler(), LogisticRegression())
    pipe.fit(embeddings, y)
    testy = pipe.predict(embeddings)
    assert testy.shape[0] == embeddings.shape[0]
