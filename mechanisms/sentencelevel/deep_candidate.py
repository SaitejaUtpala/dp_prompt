import re
import torch
import functools
import operator
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from typing import List, Optional, Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from pandarallel import pandarallel


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.cluster import KMeans

pandarallel.initialize(progress_bar=True)
CLEANR = re.compile("<.*?>")


def split_into_sentences(doc: str) -> List[str]:
    return sent_tokenize(re.sub(CLEANR, '', doc))


def preprocess_df(data_df: pd.DataFrame, field: str) -> pd.Series:
    return data_df[field].apply(split_into_sentences)


def encode(
    data_df: pd.DataFrame,
    model: torch.nn.Module,
    encode_field: str,
    device: Optional[str] = "cuda",
    mean: Optional[bool] = True,
) -> None:
    sent_field_name = f"{encode_field}_sentences"
    data_df[sent_field_name] = preprocess_df(data_df=data_df, field=encode_field)
    list_of_sentences = functools.reduce(operator.iconcat, data_df[sent_field_name], [])

    embeddings = model.encode(
        list_of_sentences, batch_size=int(32), device=device, convert_to_tensor=True
    )
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
    assert (data_df[indexing_field_name].apply(lambda s: len(embeddings[s[0] : s[1]])) == data_df_len).any()


def encode_df(
    data_df: pd.DataFrame,
    encode_fields: List[str],
    model_name: Optional[str] = "all-mpnet-base-v2",
    mean: Optional[bool] = True
):
    model = SentenceTransformer(model_name)

    for encode_field in encode_fields:
        print(f"encoding for {encode_field}")
        encode(data_df=data_df, model=model, encode_field=encode_field, mean=mean)
        
        
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_p=0.50):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    
class SentenceRecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_clusters, dropout_p=0.50):
        super(SentenceRecoder, self).__init__()
        self.encoder = MLP(input_dim, 2*input_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_clusters)
        
    def forward(self, x):
        return self.classifier(self.encoder(x))
    
    def encode(self, x):
        return self.encoder(x)
        
        
    

def train_step(model, optimizer, criterion, train_dataloader):
    model.train()
    epoch_train_loss = 0.0
    for i, (features, label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    epoch_train_loss = epoch_train_loss / len(train_dataloader)
    return epoch_train_loss


def val_step(model, val_dataloader):
    model.eval()

    all_predicted = []
    all_true = []
    with torch.no_grad():
        for i, (features, label) in enumerate(val_dataloader):
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            all_predicted.extend(predicted.cpu().numpy())
            all_true.extend(label.cpu().numpy())

        f1 = f1_score(all_true, all_predicted, average="macro")
        return f1


def train_model(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    train_epochs: Optional[int] = 50,
    print_every: Optional[int] = 1,
    verbose: Optional[bool] = False,
) -> Tuple[torch.nn.Module, List[float], List[float]]:
    
    train_losses = []
    val_f1s = []
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.90)

    best_val_f1 = 0.0
    best_weights = model.state_dict()

    val_f1 = val_step(model, val_dataloader)
    val_f1s.append(val_f1)

    if verbose:
        iterator = tqdm(range(train_epochs))
    else:
        iterator = range(train_epochs)

    for epoch in iterator:
        train_loss = train_step(model, optimizer, criterion, train_dataloader)
        train_losses.append(train_loss)
        val_f1 = val_step(model, val_dataloader)
        val_f1s.append(val_f1)

        if verbose and (epoch + 1) % print_every == 0:
            print(
                f"Epoch [{epoch+1}/{train_epochs}], Train Loss: {train_loss:.4f}, Validation F1 score: {val_f1:.4f}"
            )

        # save the model with the best validation f1 score.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = model.state_dict()

        scheduler.step()
        if verbose:
            print("lr", scheduler.get_last_lr())
    model.load_state_dict(best_weights)
    return model, train_losses, val_f1s


def get_train_mlp(
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    num_classes: int,
    input_size: Optional[int] = 768,
    hidden_size: Optional[int] = 768,
    train_epochs: Optional[int] = 50,
    device: str = "cuda",
    plot: Optional[str] = False,
    verbose: Optional[str] = False,
):
    def plot_loss(losses: List[float], label: str) -> None:
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(label)
        plt.show()

    model = MLP(input_size, hidden_size, num_classes).to(device)
    trained_model, train_losses, val_losses = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_epochs=train_epochs,
        verbose=verbose,
    )
    if plot:
        plot_loss(train_losses, label="Training Loss curve")
        plot_loss(val_losses, label="Val Acc curve")
    return trained_model, train_losses, val_losses

def get_train_sentence_recoder(
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    num_clusters: int,
    input_size: Optional[int] = 768,
    latent_dim: Optional[int] = 768,
    train_epochs: Optional[int] = 50,
    device: str = "cuda",
    plot: Optional[str] = False,
    verbose: Optional[str] = False,
):
    def plot_loss(losses: List[float], label: str) -> None:
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(label)
        plt.show()

    model = SentenceRecoder(
        input_dim=input_size, latent_dim=latent_dim, num_clusters=num_clusters
    ).to(device)
    trained_model, train_losses, val_losses = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_epochs=train_epochs,
        verbose=verbose,
    )
    if plot:
        plot_loss(train_losses, label="Training Loss curve")
        plot_loss(val_losses, label="Val Acc curve")
    return trained_model, train_losses, val_losses



def get_loader(
    data_df: pd.DataFrame,
    embedding_field: str,
    label_field: str,
    batch_size: Optional[int] = 32,
    device: Optional[str] = "cuda",
    shuffle: Optional[bool] = True,
) -> torch.utils.data.DataLoader:
    features = torch.vstack(data_df[embedding_field].tolist())
    labels = torch.tensor(data_df[label_field].tolist(), dtype=torch.long).to(device)
    features.requires_grad = False
    labels.requires_grad = False
    assert len(features) == len(labels)
    dataset = torch.utils.data.TensorDataset(features, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )
    return dataloader


import gc 
import torch
from torch.testing import assert_close
from torchtyping import TensorType
from typing import Tuple
import diffprivlib


def projections(
    cands: TensorType["num_cands", "embed_dim"],
    sents: TensorType["num_sents", "embed_dim"],
    p: int,
    device:str = 'cuda',
) -> Tuple[TensorType["num_cands", "p"], TensorType["num_sents", "p"]]:
    """
    Project candidates and sentences onto random directions.

    Args:
        cands (TensorType["num_cands", "embed_dim"]): Tensor of candidate embeddings.
        sents (TensorType["num_sents", "embed_dim"]): Tensor of sentence embeddings.
        p (int): Number of random directions for projection.
        device (str): Device to use (default is 'cuda').

    Returns:
        Tuple[TensorType["num_cands", "p"], TensorType["num_sents", "p"]]: Projected candidates and sentences.
    """
    d = sents.shape[-1]  # [k, d]
    vec = torch.normal(0, 1, size=(p, d)).to(device)  # [p, d]
    #now vecs are random vectors on sphere.
    vec = F.normalize(vec,dim=-1)
    proj_cands = cands @ vec.T  # [m, p]
    proj_sents = sents @ vec.T  # [k, p]

    return proj_cands, proj_sents


def candidates_depth(
    proj_cands: TensorType["num_cands", "p"], proj_sents: TensorType["num_sents", "p"]
) -> Tuple[TensorType["num_cands"]]:
    """
    Calculate the depth of candidates using projections.

    Args:
        proj_cands (TensorType["num_cands", "p"]): Tensor of candidate projections.
        proj_sents (TensorType["num_sents", "p"]): Tensor of sentence projections.

    Returns:
        Tuple[TensorType["num_cands"]]: Depth of each candidate.
    """
    m, p = proj_cands.shape
    k, p = proj_sents.shape

    bool_tensor_ilj = torch.ge(
        proj_sents.unsqueeze(0), proj_cands.unsqueeze(1)
    )  # [m,k,p]
    assert bool_tensor_ilj.shape == (m, k, p)
    hij = bool_tensor_ilj.sum(1)  # [m,p]
    depth_i = hij.min(axis=-1)[0]  # type: ignore #[m]
    return depth_i


def approx_tukey_depth(
    cands: TensorType["num_cands", "embed_dim"],
    sents: TensorType["num_sents", "embed_dim"],
    p: int,
) -> TensorType["num_cands"]:
    """
    Calculate an approximation of Tukey depth for each candidate with respect to sentences.

    Args:
        cands (TensorType["num_cands", "embed_dim"]): Tensor of candidate embeddings.
        sents (TensorType["num_sents", "embed_dim"]): Tensor of sentence embeddings.
        p (int): Parameter for the approximation.

    Returns:
        TensorType["num_cands"]: Depth of each candidate.
    """
    proj_cands, proj_sents = projections(cands=cands, sents=sents, p=p)
    depth = candidates_depth(proj_cands=proj_cands, proj_sents=proj_sents)
    return depth


def private_maximizer(scores, epsilon, sensitivity=1.0):
    """
    Apply the Exponential mechanism to privatize the maximum value in the scores.

    Args:
        scores: Scores for which to find the maximum.
        epsilon: Privacy parameter.
        sensitivity: Sensitivity of the function (default is 1.0).

    Returns:
        int: Privatized index of the maximum score.
    """
    mech = diffprivlib.mechanisms.Exponential(epsilon=epsilon,sensitivity=sensitivity, utility=scores.cpu().tolist())
    private_max_idx = mech.randomise()
    return private_max_idx

        
@torch.inference_mode
def get_recoded_embeddings(trained_sentence_recoder, embeddings):
    """
    Get recoded embeddings using a trained sentence recoder model.

    Args:
        trained_sentence_recoder: Trained sentence recoder model.
        embeddings: Original embeddings to be recoded.

    Returns:
        torch.Tensor: Recoded embeddings.
    """
    trained_sentence_recoder.eval()
    recoded_embeddings = trained_sentence_recoder.encode(embeddings)
    assert recoded_embeddings.shape == embeddings.shape
    return recoded_embeddings





def set_labels(
    deep_candidate_train_df: pd.DataFrame,
    deep_candidate_val_df: pd.DataFrame,
    num_clusters: int,
) -> None:
    """
    Assign clustering labels to deep candidate data using K-Means clustering.

    Args:
        deep_candidate_train_df (pd.DataFrame): DataFrame of deep candidate training data.
        deep_candidate_val_df (pd.DataFrame): DataFrame of deep candidate validation data.
        num_clusters (int): Number of clusters for K-Means clustering.

    Returns:
        None
    """
    
    kmeans = KMeans(n_clusters=num_clusters, verbose=0, tol=1e-5)
    clustering_data = (
        torch.vstack(
            [
                torch.vstack(deep_candidate_train_df["review_embeddings"].tolist()),
                torch.vstack(deep_candidate_val_df["review_embeddings"].to_list()),
            ]
        )
        .cpu()
        .numpy()
    )
    assert len(clustering_data) == len(deep_candidate_train_df) + len(
        deep_candidate_val_df
    )
    clustering_labels = kmeans.fit_predict(clustering_data)
    deep_candidate_train_df["clustering_label"] = clustering_labels[
        : len(deep_candidate_train_df)
    ]
    deep_candidate_val_df["clustering_label"] = clustering_labels[
        len(deep_candidate_train_df) :
    ]


def get_sentence_recoder(
    deep_candidate_train_df: pd.DataFrame,
    deep_candidate_val_df: pd.DataFrame,
    num_clusters: int,
    train_epochs: Optional[int] = 100,
    verbose: Optional[bool] = False,
    plot: Optional[bool] = False,
) -> torch.nn.Module:
    """
    Train a sentence recoder model for clustering deep candidates.

    Args:
        deep_candidate_train_df (pd.DataFrame): DataFrame of deep candidate training data.
        deep_candidate_val_df (pd.DataFrame): DataFrame of deep candidate validation data.
        num_clusters (int): Number of clusters for the sentence recoder.
        train_epochs (Optional[int]): Number of training epochs (default is 100).
        verbose (Optional[bool]): Whether to display training progress (default is False).
        plot (Optional[bool]): Whether to plot training progress (default is False).

    Returns:
        torch.nn.Module: Trained sentence recoder model.
    """
    
    set_labels(
        deep_candidate_train_df=deep_candidate_train_df,
        deep_candidate_val_df=deep_candidate_val_df,
        num_clusters=num_clusters,
    )

    assert "clustering_label" in deep_candidate_train_df.columns
    assert "clustering_label" in deep_candidate_val_df.columns

    train_dataloader = get_loader(
        data_df=deep_candidate_train_df,
        embedding_field="review_embeddings",
        label_field="clustering_label",
    )
    val_dataloader = get_loader(
        data_df=deep_candidate_val_df,
        embedding_field="review_embeddings",
        label_field="clustering_label",
    )
    trained_model, train_losses, val_losses = get_train_sentence_recoder(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_clusters=num_clusters,
        train_epochs=train_epochs,
        verbose=verbose,
        plot=plot,
    )
    return trained_model


def deep_candidate(
    deep_candidate_train_df: pd.DataFrame,
    deep_candidate_val_df: pd.DataFrame,
    data_df: pd.DataFrame,
    epsilon_list: List[float],
    num_clusters: Optional[int] = 50,
    verbose: Optional[bool] = False,
) -> torch.Tensor:
    """
    Perform deep candidate selection with differential privacy.

    Args:
        deep_candidate_train_df (pd.DataFrame): DataFrame with training data for deep candidates.
        deep_candidate_val_df (pd.DataFrame): DataFrame with validation data for deep candidates.
        data_df (pd.DataFrame): DataFrame with data to select candidates from.
        epsilon_list (List[float]): List of privacy parameters for differential privacy.
        num_clusters (Optional[int]): Number of clusters for sentence recoder (default is 50).
        verbose (Optional[bool]): Whether to display verbose information (default is False).

    Returns:
        torch.Tensor: Private embeddings for the selected candidates.
    """
    
    encode_fields = ["review"]
    encode_df(data_df=deep_candidate_train_df, encode_fields=encode_fields, mean=True)
    encode_df(data_df=deep_candidate_val_df, encode_fields=encode_fields, mean=True)
    encode_df(data_df=data_df, encode_fields=encode_fields, mean=False)

    assert "review_embeddings" in deep_candidate_train_df.columns
    assert "review_embeddings" in deep_candidate_val_df.columns
    assert "review_embeddings" in data_df.columns

    trained_sentence_recoder = get_sentence_recoder(
        deep_candidate_train_df=deep_candidate_train_df,
        deep_candidate_val_df=deep_candidate_val_df,
        num_clusters=num_clusters,
        verbose=verbose,
        plot=verbose,
    )

    with torch.inference_mode():
        
        recoded_cands_embeddings = get_recoded_embeddings(
            trained_sentence_recoder=trained_sentence_recoder,
            embeddings=torch.vstack(
                deep_candidate_train_df["review_embeddings"].to_list()
            ),
        )
        assert recoded_cands_embeddings.shape == (5000, 768)

        data_embeddings_list = data_df["review_embeddings"].to_list()
        private_embeddings = {epsilon: [] for epsilon in epsilon_list}
        num_projections = 100

        for ix, data_embedding in tqdm(enumerate(data_embeddings_list)):
            recoded_data_embedding = get_recoded_embeddings(
                trained_sentence_recoder=trained_sentence_recoder,
                embeddings=data_embedding,
            )
            depth = approx_tukey_depth(
                cands=recoded_cands_embeddings,
                sents=recoded_data_embedding,
                p=num_projections,
            )
            
            for epsilon in epsilon_list:
                private_idx = private_maximizer(scores=depth, epsilon=epsilon)
                private_embedding = recoded_cands_embeddings[private_idx]
                private_embeddings[epsilon].append(private_embedding)

        for epsilon in epsilon_list:
            private_embeddings[epsilon] = torch.vstack(private_embeddings[epsilon])
            assert len(private_embeddings[epsilon]) == len(data_embeddings_list)
            
    return private_embeddings