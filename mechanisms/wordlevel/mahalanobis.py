import numpy as np
import re
import nltk
from tqdm import tqdm
from gensim.models import Word2Vec

import gensim.downloader as api
from pandarallel import pandarallel
from gensim.similarities import AnnoyIndex
from nltk.tokenize import word_tokenize
from sacremoses import MosesDetokenizer
from gensim.similarities.annoy import AnnoyIndexer
import pandas as pd
from typing import List


num_trees = 500
nltk.download("punkt")
html_cleaner = re.compile("<.*?>")
pandarallel.initialize(progress_bar=True)
model_gigaword = api.load("glove-wiki-gigaword-50")
annoy_index = AnnoyIndexer(model_gigaword, num_trees)


from tqdm import tqdm


def scaled_covariance_for_mahalanobis(model: Word2Vec) -> np.ndarray:
    """
    Compute the scaled covariance matrix for Mahalanobis differential privacy.

    Args:
        model (Word2Vec): The word embedding model.

    Returns:
        np.ndarray: The scaled covariance matrix.
    """
    word_embeddings = model.vectors
    sample_covariance_matrix = np.cov(word_embeddings.T)
    variance = np.var(word_embeddings, axis=0)
    mean_variance = variance.mean()
    scaled_covariance_matrix = sample_covariance_matrix / mean_variance

    # check the trace is equal np identity
    assert np.allclose(
        np.trace(scaled_covariance_matrix), word_embeddings.shape[-1], atol=1e-1
    )
    return scaled_covariance_matrix


def multivariate_mahalanobis_sample(
    reg_cov_sq_root: np.ndarray, dimension: int, epsilon: float
) -> np.ndarray:
    """
    Generate a multivariate Mahalanobis noise sample.

    Args:
        reg_cov_sq_root (np.ndarray): The square root of the scaled covariance matrix.
        dimension (int): The dimension of the noise vector.
        epsilon (float): The privacy parameter.

    Returns:
        np.ndarray: A noise vector sampled from a multivariate Mahalanobis distribution.
    """
    rand_vec = np.random.normal(size=dimension)
    normalized_vec = rand_vec / np.linalg.norm(rand_vec)
    magnitude = np.random.gamma(shape=dimension, scale=1 / epsilon)
    return magnitude * (reg_cov_sq_root @ normalized_vec)


def mahalanobis(
    review: str,
    model: Word2Vec,
    html_cleaner: Any,
    indexer: AnnoyIndex,
    reg_cov_sq_root: np.ndarray,
    epsilon: float,
) -> str:
    """
    Apply differential privacy using the Mahalanobis mechanism to a text review.

    Args:
        review (str): The input text review.
        model (Word2Vec): The word embedding model.
        html_cleaner (Any): Regular expression pattern for cleaning HTML tags.
        indexer (AnnoyIndex): Annoy index for efficient nearest neighbor search.
        reg_cov_sq_root (np.ndarray): The square root of the scaled covariance matrix.
        epsilon (float): The privacy parameter.

    Returns:
        str: The differentially private version of the input review.
    """
    dimension = model_gigaword.vectors.shape[-1]
    review = word_tokenize(re.sub(html_cleaner, "", review.lower()))
    priv_words = []

    for word in review:
        if word in model.key_to_index:
            v = model[word]
            per = v + multivariate_mahalanobis_sample(
                reg_cov_sq_root=reg_cov_sq_root, dimension=dimension, epsilon=epsilon
            )
            priv_word = model.most_similar([per], topn=1, indexer=indexer)[0][0]
            priv_words.append(priv_word)

    priv_review = MosesDetokenizer().detokenize(priv_words, return_str=True)
    return priv_review
