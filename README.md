# Clearformers

Clearformers help create topic-based embeddings as described by my Bsc-thesis. It follows an sklearn-style API and should be relatively easy to use :))

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install clearformers. (NB: it is not on pypi, yet!)

```bash
git clone https://github.com/Rysias/clearformers.git
cd clearformers
pip install .
```

## Usage

```python
import numpy as np
from bertopic import BERTopic
from explainlp.clearformers import Clearformer

topic_model = BERTopic()
clearformer = Clearformer(topic_model)

embeddings = np.random.rand(50, 756)
topic_embeddings = clearformer.fit_transform(embeddings)

```

## License
[MIT](https://choosealicense.com/licenses/mit/)