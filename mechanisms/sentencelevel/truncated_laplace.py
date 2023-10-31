import re
import gc
import torch
import functools
import operator
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from typing import List, Optional, Dict, List, Tuple
from sentence_transformers import SentenceTransformer


def split_into_sentences(doc: str) -> List[str]:
    """
    Split a document into sentences.

    Args:
        doc (str): The input document.

    Returns:
        List[str]: A list of sentences.
    """
    return sent_tokenize(re.sub(CLEANR, "", doc))


def preprocess_df(data_df: pd.DataFrame, field: str) -> pd.Series:
    """
    Preprocess a DataFrame field by splitting it into sentences.

    Args:
        data_df (pd.DataFrame): The input DataFrame.
        field (str): The name of the field to preprocess.

    Returns:
        pd.Series: A Series containing the preprocessed sentences.
    """
    return data_df[field].apply(split_into_sentences)


def encode(
    data_df: pd.DataFrame,
    model: torch.nn.Module,
    encode_field: str,
    device: Optional[str] = "cuda",
    mean: Optional[bool] = True,
    convert_to_tensor: Optional[bool] = True,
) -> None:
    """
    Encode sentences in a DataFrame field and store the embeddings.

    Args:
        data_df (pd.DataFrame): The input DataFrame.
        model (torch.nn.Module): The SentenceTransformer model for encoding.
        encode_field (str): The name of the field to encode.
        device (Optional[str]): The device to use (default is "cuda").
        mean (Optional[bool]): Whether to calculate the mean embedding (default is True).
        convert_to_tensor (Optional[bool]): Whether to convert embeddings to tensors (default is True).

    Returns:
        None
    """

    sent_field_name = f"{encode_field}_sentences"
    data_df[sent_field_name] = preprocess_df(data_df=data_df, field=encode_field)
    list_of_sentences = functools.reduce(operator.iconcat, data_df[sent_field_name], [])

    embeddings = model.encode(
        list_of_sentences,
        batch_size=int(32),
        device=device,
        convert_to_tensor=convert_to_tensor,
    )
    if convert_to_tensor:
        embeddings.requires_grad = False

    data_df_len = data_df[sent_field_name].apply(len)
    data_df_cumsum = data_df_len.cumsum()
    indexing_field_name = f"{encode_field}_indexing"
    data_df[indexing_field_name] = list(
        zip(data_df_cumsum - data_df_len, data_df_cumsum)
    )
    embeddings_field_name = f"{encode_field}_embeddings"
    if mean:
        data_df[embeddings_field_name] = data_df[indexing_field_name].apply(
            lambda s: embeddings[s[0] : s[1]].mean(axis=0)
        )
    else:
        data_df[embeddings_field_name] = data_df[indexing_field_name].apply(
            lambda s: embeddings[s[0] : s[1]]
        )

    # sanity checks
    assert (
        data_df[indexing_field_name].apply(lambda s: len(embeddings[s[0] : s[1]]))
        == data_df_len
    ).any()


def encode_df(
    data_df: pd.DataFrame,
    encode_fields: List[str],
    model_name: Optional[str] = "all-mpnet-base-v2",
    mean: Optional[bool] = True,
    convert_to_tensor: Optional[bool] = True,
):
    """
    Encode sentences in a DataFrame for specified fields.

    Args:
        data_df (pd.DataFrame): The input DataFrame.
        encode_fields (List[str]): List of field names to encode.
        model_name (Optional[str]): The name of the SentenceTransformer model (default is "all-mpnet-base-v2").
        mean (Optional[bool]): Whether to calculate the mean embedding (default is True).
        convert_to_tensor (Optional[bool]): Whether to convert embeddings to tensors (default is True).

    Returns:
        None
    """
    model = SentenceTransformer(model_name)

    for encode_field in encode_fields:
        print(f"encoding for {encode_field}")
        encode(
            data_df=data_df,
            model=model,
            encode_field=encode_field,
            mean=mean,
            convert_to_tensor=convert_to_tensor,
        )


def get_clipping_boundaries(
    deep_candidates: np.array, thresh_pct: float = 0.75
) -> Tuple[np.array, np.array, np.array]:
    """
    Calculate clipping boundaries based on deep candidate embeddings.

    Args:
        deep_candidates: Numpy array of deep candidate embeddings.
        thresh_pct (float): Threshold percentage (default is 0.75).

    Returns:
        Tuple[np.array, np.array, np.array]: Lower thresholds, upper thresholds, and sensitivities.
    """
    thresh_n = int(thresh_pct * len(deep_candidates))

    thresholds = []
    medians = np.median(deep_candidates, axis=0)
    dist_from_med = np.abs(deep_candidates - medians)
    for dim in range(deep_candidates.shape[1]):
        idx_closest_to_med = np.argsort(dist_from_med[:, dim])[:thresh_n]
        pts_closest_to_med = deep_candidates[idx_closest_to_med, dim]
        min_thresh = np.min(pts_closest_to_med)
        max_thresh = np.max(pts_closest_to_med)
        thresholds.append((min_thresh, max_thresh))

    upper_thresholds = np.array([thresh[1] for thresh in thresholds])
    lower_thresholds = np.array([thresh[0] for thresh in thresholds])
    sensitivities = upper_thresholds - lower_thresholds

    return lower_thresholds, upper_thresholds, sensitivities


def clip(
    data_embeddings: np.array, lower_thresholds: np.array, upper_thresholds: np.array
) -> np.array:
    """
    return a clipped embedding with each dimension truncated to the the
    hypercube (cube? rectangular prism?) defined by thresholds.

    Inputs:
        embd: (n x d) array of n d-dimensional embeddings
        lower_thresholds: list of d lower thresholds
        upper_thresholds: list of d upper thresholds
    Outputs:
        embd_clipped: list of clipped d-dimensional embeddings
    """
    too_big = data_embeddings > upper_thresholds
    too_small = data_embeddings < lower_thresholds
    clipped_data_embeddings = data_embeddings.copy()
    for dim in range(data_embeddings.shape[1]):
        clipped_data_embeddings[too_big[:, dim], dim] = upper_thresholds[dim]
        clipped_data_embeddings[too_small[:, dim], dim] = lower_thresholds[dim]

    return clipped_data_embeddings


def laplace_mechanism(
    data_embedding: np.array, sensitivities: np.array, epsilon: float
) -> np.array:
    """
    Apply Laplace mechanism for differential privacy.

    Args:
        data_embedding: Numpy array of data embeddings.
        sensitivities: Numpy array of sensitivities.
        epsilon: Privacy parameter.

    Returns:
        np.array: Privatized embeddings.
    """
    k_sentences = len(data_embedding)
    mean_sentence_embedding = data_embedding.mean(axis=0, keepdims=True)
    assert mean_sentence_embedding.shape == (1, 768)

    noise = np.zeros(mean_sentence_embedding.shape)
    dim = mean_sentence_embedding.shape[1]
    eps_effective = k_sentences * epsilon / dim

    for dim in range(dim):
        noise[:, dim] = np.random.laplace(
            scale=sensitivities[dim] / eps_effective, size=1
        )

    assert noise.shape == mean_sentence_embedding.shape
    private_embedding = (mean_sentence_embedding + noise)[0]
    assert private_embedding.shape == (768,)
    return private_embedding


def truncated_laplace(
    deep_candidate_train_df: pd.DataFrame,
    deep_candidate_val_df: pd.DataFrame,
    data_df: pd.DataFrame,
    epsilon_list: List[float],
    thresh_pct: Optional[float] = 0.75,
) -> torch.Tensor:
    """
    Apply truncated Laplace mechanism to embeddings.

    Args:
        deep_candidate_train_df (pd.DataFrame): DataFrame of deep candidate training data.
        deep_candidate_val_df (pd.DataFrame): DataFrame of deep candidate validation data.
        data_df (pd.DataFrame): Main data DataFrame.
        epsilon_list (List[float]): List of epsilon values.
        thresh_pct (Optional[float]): Threshold percentage (default is 0.75).

    Returns:
        torch.Tensor: Privatized embeddings.
    """
    encode_fields = ["review"]
    encode_df(
        data_df=deep_candidate_train_df,
        encode_fields=encode_fields,
        mean=True,
        convert_to_tensor=False,
    )
    encode_df(
        data_df=deep_candidate_val_df,
        encode_fields=encode_fields,
        mean=True,
        convert_to_tensor=False,
    )
    encode_df(
        data_df=data_df,
        encode_fields=encode_fields,
        mean=False,
        convert_to_tensor=False,
    )

    assert "review_embeddings" in deep_candidate_train_df.columns
    assert "review_embeddings" in deep_candidate_val_df.columns
    assert "review_embeddings" in data_df.columns

    deep_candidates = np.vstack(
        deep_candidate_train_df["review_embeddings"].to_list()
        + deep_candidate_val_df["review_embeddings"].to_list()
    )
    lower_thresholds, upper_thresholds, sensitivities = get_clipping_boundaries(
        deep_candidates=deep_candidates, thresh_pct=thresh_pct
    )
    print("a", lower_thresholds.shape)
    print("b", upper_thresholds.shape)
    print("c", sensitivities.shape)

    assert upper_thresholds.shape == lower_thresholds.shape == sensitivities.shape
    data_embeddings_list = data_df["review_embeddings"].to_list()
    private_embeddings = {epsilon: [] for epsilon in epsilon_list}

    for ix, data_embedding in tqdm(enumerate(data_embeddings_list)):
        clipped_data_embeddings = clip(
            data_embedding,
            lower_thresholds=lower_thresholds,
            upper_thresholds=upper_thresholds,
        )

        for epsilon in epsilon_list:
            private_embedding = laplace_mechanism(
                data_embedding=clipped_data_embeddings,
                sensitivities=sensitivities,
                epsilon=epsilon,
            )

            private_embeddings[epsilon].append(private_embedding)

    for epsilon in epsilon_list:
        private_embeddings[epsilon] = np.vstack(private_embeddings[epsilon])
        assert len(private_embeddings[epsilon]) == len(data_embeddings_list)

    return private_embeddings
