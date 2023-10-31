import numpy as np
import re
import nltk
from tqdm import tqdm
from gensim.models import Word2Vec

import gensim.downloader as api
from pandarallel import pandarallel
from nltk.tokenize import word_tokenize
from sacremoses import MosesDetokenizer
from gensim.similarities.annoy import AnnoyIndexer
import pandas as pd
from typing import List, Any


num_trees = 500
nltk.download("punkt")
html_cleaner = re.compile("<.*?>")
pandarallel.initialize(progress_bar=True)
model_gigaword = api.load("glove-wiki-gigaword-50")
annoy_index = AnnoyIndexer(model_gigaword, num_trees)

def multivariate_laplace(dimension: int, epsilon: float) -> np.ndarray:
    """
    Generate a multivariate Laplace noise sample.

    Args:
        dimension (int): The dimension of the noise vector.
        epsilon (float): The privacy parameter.

    Returns:
        np.ndarray: A noise vector sampled from a multivariate Laplace distribution.
    """
    rand_vec = np.random.normal(size=dimension)
    normalized_vec = rand_vec / np.linalg.norm(rand_vec)
    magnitude = np.random.gamma(shape=dimension, scale=1 / epsilon)
    return normalized_vec * magnitude


def madlib(
    review: str, model: Word2Vec, html_cleaner: str, indexer: AnnoyIndexer, epsilon: float
) -> str:
    """
    Apply madlib mechansim to a single text review.

    Args:
        review (str): The input text review.
        model (Word2Vec): The word embedding model.
        html_cleaner (str): Regular expression pattern for cleaning HTML tags.
        indexer (AnnoyIndex): Annoy index for efficient nearest neighbor search.
        epsilon (float): The privacy parameter.

    Returns:
        str: The differentially private version of the input review.
    """
    dimension = model.vectors.shape[-1]
    review = word_tokenize(re.sub(html_cleaner, "", review.lower()))
    priv_words = []
    for word in review:
        if word in model.key_to_index:
            v = model[word]
            per = v + multivariate_laplace(dimension=dimension, epsilon=epsilon)
            priv_word = model.most_similar([per], topn=1, indexer=indexer)[0][0]
            priv_words.append(priv_word)
    priv_review = MosesDetokenizer().detokenize(priv_words, return_str=True)
    return priv_review


def madlib_df(
    eps_list: List[float], df: pd.DataFrame, field: str, save_path: str
) -> pd.DataFrame:
    """
    Apply differential privacy to a list of text reviews with varying epsilon values and save the results to a CSV file.

    Args:
        eps_list (List[float]): A list of epsilon values to apply differential privacy.
        df (pd.DataFrame): The DataFrame containing the text reviews.
        field (str): The name of the field containing the text reviews.
        save_path (str): The path to save the results CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the differentially private versions of text reviews.
    """
    review_list = []
    for eps in tqdm(eps_list):
        x = df[field].parallel_apply(
            lambda review: madlib(
                review=review,
                model=model_gigaword,
                html_cleaner=html_cleaner,
                indexer=annoy_index,
                epsilon=eps,
            )
        )
        review_list.append(x)

    madlib_df = pd.DataFrame(
        zip(*review_list), columns=[f"madlib_eps={eps}" for eps in eps_list]
    )
    madlib_df.to_csv(save_path, index=False)
    return madlib_df
