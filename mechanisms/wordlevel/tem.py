import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from sacremoses import MosesDetokenizer
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from typing import Tuple


from gensim.models import Word2Vec

import pandas as pd
from typing import List

import nltk

nltk.download("punkt")
beta = 0.001
html_cleaner = re.compile("<.*?>")
model_gigaword = api.load("glove-wiki-gigaword-50")


def get_threshold(beta: float, epsilon: float, model: Word2Vec) -> float:
    """
    Calculate the threshold (gamma) for the Truncated Exponential Mechanism.

    Args:
        beta (float): The privacy parameter beta.
        epsilon (float): The privacy parameter epsilon.
        model (Word2Vec): The word embedding model.

    Returns:
        float: The threshold (gamma) for the Truncated Exponential Mechanism.
    """
    vocab_size = len(model.vectors)
    x = ((1 - beta) / beta) * (vocab_size - 1)
    gamma = (2 / epsilon) * np.log(x)
    return gamma


def radius_neighbors(
    vectors: Tensor, vector: Tensor, radius: float
) -> Tuple[Tensor, Tensor]:
    """
    Find neighbors within a given radius of a target vector in a set of vectors.

    Args:
        vectors (Tensor): The set of vectors to search within.
        vector (Tensor): The target vector for which neighbors are sought.
        radius (float): The search radius to find neighbors.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing two tensors:
            - dists (Tensor): Distances of the neighbors from the target vector.
            - ids (Tensor): Indices of the neighbors in the input vectors.
    """
    x = torch.linalg.norm(vectors - vector, axis=-1)
    in_radius = x < radius
    dists, ids = x[in_radius], (in_radius).nonzero()[:, 0]
    return dists, ids


def truncated_exponential_mechanism(
    word: str,
    model: Word2Vec,
    cuda_vectors: torch.Tensor,
    gumbel_sampler: torch.distributions.Gumbel,
    gamma: float,
    epsilon: float,
):
    """
    Apply the truncated exponential mechanism to generate a differentially private word.

    Args:
        word (str): The input word.
        model (gensim.models.keyedvectors.Word2VecKeyedVectors): The word embedding model.
        cuda_vectors (torch.Tensor): Word embeddings as torch Tensor on GPU.
        gumbel_sampler (torch.distributions.Gumbel): Gumbel noise sampler.
        gamma (float): The threshold for the truncated exponential mechanism.
        epsilon (float): The privacy parameter.

    Returns:
        str: The differentially private word.
    """

    vector = cuda_vectors[model.key_to_index[word], None]
    dists, ids = radius_neighbors(cuda_vectors, vector, gamma)
    assert dists.shape == ids.shape

    threshold = gamma
    vocab_size = len(model.vectors)
    out_of_threshold_size = torch.tensor([vocab_size - len(dists)])

    perp_score = -1 * threshold + 2 * (torch.log(out_of_threshold_size) / epsilon)
    scores = -1 * dists
    scores = torch.hstack([scores, perp_score.cuda()])

    noises = gumbel_sampler.sample((len(dists) + 1,))[:, 0]
    assert noises.shape == scores.shape

    noisy_scores = scores + noises
    max_idx = torch.argmax(noisy_scores).item()

    if max_idx == len(noisy_scores) - 1:
        support = np.setdiff1d(np.arange(vocab_size), ids.cpu().numpy())
        assert len(support) == out_of_threshold_size
        private_word_id = np.random.choice(support)
    else:
        private_word_id = ids[max_idx]
    private_word = model.index_to_key[private_word_id]
    return private_word


def tem_review(
    review: str,
    model: Word2Vec,
    cuda_vectors: torch.Tensor,
    gumbel_sampler: torch.distributions.Gumbel,
    gamma: float,
    html_cleaner: re.Pattern,
    epsilon: float,
):
    """
    Apply the Truncated Exponential Mechanism (TEM) to a text review to generate a differentially private version.

    Args:
        review (str): The input text review.
        model (gensim.models.keyedvectors.Word2VecKeyedVectors): The word embedding model.
        cuda_vectors (torch.Tensor): Word embeddings as a torch Tensor on GPU.
        gumbel_sampler (torch.distributions.Gumbel): Gumbel noise sampler.
        gamma (float): The threshold for the truncated exponential mechanism.
        html_cleaner (re.Pattern): Regular expression pattern for cleaning HTML tags.
        epsilon (float): The privacy parameter.

    Returns:
        str: The differentially private version of the input review.
    """
    review = word_tokenize(re.sub(html_cleaner, "", review.lower()))
    priv_words = []

    for word in review:
        if word in model.key_to_index:
            priv_word = truncated_exponential_mechanism(
                word=word,
                model=model,
                cuda_vectors=cuda_vectors,
                gumbel_sampler=gumbel_sampler,
                gamma=gamma,
                epsilon=epsilon,
            )
            priv_words.append(priv_word)

    priv_review = MosesDetokenizer().detokenize(priv_words, return_str=True)
    return priv_review


def tem_df(
    df: pd.DataFrame, model: Word2Vec, epsilon_list: List[float], save_path: str
) -> pd.DataFrame:
    """
    Apply the Truncated Exponential Mechanism (TEM) to a DataFrame of text reviews with varying epsilon values.

    Args:
        df (pd.DataFrame): The DataFrame containing the text reviews.
        model (gensim.models.keyedvectors.Word2VecKeyedVectors): The word embedding model.
        epsilon_list (List[float]): A list of epsilon values for differential privacy.
        save_path (str): The path to save the results CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the differentially private versions of text reviews.
    """

    cuda_vectors = torch.from_numpy(model.vectors).float().cuda()
    private_df_list = []
    for epsilon in epsilon_list:
        gamma = get_threshold(beta=beta, epsilon=epsilon, model=model)
        mean = torch.tensor([0.0]).cuda()
        scale = torch.tensor([2 / epsilon]).cuda()
        gumbel_sampler = torch.distributions.gumbel.Gumbel(mean, scale)

        private_df_eps = df["review"].progress_apply(
            lambda review: tem_review(
                review=review,
                model=model,
                cuda_vectors=cuda_vectors,
                gumbel_sampler=gumbel_sampler,
                gamma=gamma,
                html_cleaner=html_cleaner,
                epsilon=epsilon,
            )
        )
        private_df_list.append(private_df_eps)

    private_df = pd.DataFrame(
        zip(*private_df_list),
        columns=[f"tem_eps={epsilon}" for epsilon in epsilon_list],
    )

    private_df.to_csv(save_path, index=False)
    return private_df
