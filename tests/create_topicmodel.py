from pathlib import Path
import random
from typing import Any
import numpy as np
import pickle
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups


def dump_pickle(obj: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"]
random.seed(10)
small_docs = random.choices(docs, k=500)
topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(small_docs)

embeddings = topic_model._extract_embeddings(small_docs, verbose=True)

topic_model.save(str(Path("tests/topic_model")))
np.save(Path("tests/example_embeddings.npy"), embeddings)
dump_pickle(small_docs, Path("tests/example_docs.pkl"))
