import numpy as np

from cuanalytics.similarity import euclidean, manhattan, cosine, jaccard


def test_euclidean():
    assert euclidean([0, 0], [3, 4]) == 5.0


def test_manhattan():
    assert manhattan([1, 2], [4, 6]) == 7.0


def test_cosine():
    assert np.isclose(cosine([1, 0], [0, 1]), 1.0)


def test_jaccard():
    assert np.isclose(jaccard([1, 0, 1], [1, 1, 0]), 2/3)
