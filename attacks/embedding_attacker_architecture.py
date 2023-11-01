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
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score
import torch.optim.lr_scheduler as lr_scheduler


pandarallel.initialize(progress_bar=True)


def split_into_sentences(doc: str) -> List[str]:
    """
    Split a text document into a list of sentences.

    Args:
        doc (str): The input text document.

    Returns:
        List[str]: A list of sentences extracted from the input document.
    """
    try:
        return sent_tokenize(doc)
    except Exception as e:
        print("yikes")
        print(type(doc))
        print(doc)
        return [""]


def preprocess_df(data_df: pd.DataFrame, field: str) -> pd.Series:
    """
    Preprocess text data in a DataFrame by splitting it into sentences.

    Args:
        data_df (pd.DataFrame): The DataFrame containing text data to be preprocessed.
        field (str): The name of the field in the DataFrame to be processed.

    Returns:
        pd.Series: A Series containing the preprocessed data with text split into sentences.
    """
    return data_df[field].parallel_apply(split_into_sentences)


def encode(
    data_df: pd.DataFrame,
    model: torch.nn.Module,
    encode_field: str,
    device: Optional[str] = "cuda",
) -> None:
    """
    Encode text data in a DataFrame using a pre-trained model.

    Args:
        data_df (pd.DataFrame): The DataFrame containing text data to be encoded.
        model (torch.nn.Module): The pre-trained model for encoding text.
        encode_field (str): The name of the field in the DataFrame to be encoded.
        device (Optional[str]): The device (e.g., "cuda" for GPU) to use for encoding. Default is "cuda."

    Returns:
        None
    """
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
    data_df[embeddings_field_name] = data_df[indexing_field_name].apply(
        lambda s: embeddings[s[0] : s[1]].mean(axis=0)
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
):
    """
    Encode text data in a DataFrame using a pre-trained model from SentenceTransformer.

    Args:
        data_df (pd.DataFrame): The DataFrame containing text data to be encoded.
        encode_fields (List[str]): List of field names in the DataFrame to be encoded.
        model_name (Optional[str]): Name of the pre-trained model from SentenceTransformer.
            Default is "all-mpnet-base-v2."

    Returns:
        None
    """
    model = SentenceTransformer(model_name)

    for encode_field in encode_fields:
        print(f"encoding for {encode_field}")
        encode(data_df=data_df, model=model, encode_field=encode_field)


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


def train_step(model, optimizer, criterion, train_dataloader):
    """
    Perform a training step for a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        optimizer (torch.optim.Optimizer): The optimizer to update model parameters.
        criterion (nn.Module): The loss criterion used for training.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.

    Returns:
        float: The average training loss for the current training step.
    """
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
    """
    Perform a validation step for a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.

    Returns:
        float: The macro F1 score for the validation data.
    """

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
    train_epochs: Optional[int] = 100,
    print_every: Optional[int] = 1,
    verbose: Optional[bool] = False,
) -> Tuple[torch.nn.Module, List[float], List[float]]:
    """
    Train a PyTorch model on training data and validate it on validation data.

    Args:
        model (torch.nn.Module): The PyTorch model to train and validate.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation data.
        train_epochs (Optional[int]): Number of training epochs. Default is 100.
        print_every (Optional[int]): Print progress every `print_every` epochs. Default is 1.
        verbose (Optional[bool]): If True, print training progress.

    Returns:
        Tuple[torch.nn.Module, List[float], List[float]]: A tuple containing the trained model,
        a list of training loss values, and a list of validation F1 score values.
    """

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
    train_epochs: Optional[int] = 20,
    device: str = "cuda",
    plot: Optional[str] = False,
    verbose: Optional[str] = False,
):
    """
    Create and train a multi-layer perceptron (MLP) model for classification.

    Args:
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation data.
        num_classes (int): Number of classes for classification.
        input_size (Optional[int]): Size of the input features. Default is 768.
        hidden_size (Optional[int]): Size of the hidden layer in the MLP. Default is 768.
        train_epochs (Optional[int]): Number of training epochs. Default is 20.
        device (str): The device to use for training (e.g., "cuda" for GPU).
        plot (Optional[bool]): If True, plot training and validation loss curves.
        verbose (Optional[bool]): If True, print training progress.

    Returns:
        Tuple[torch.nn.Module, List[float], List[float]]: A tuple containing the trained MLP model,
        a list of training loss values, and a list of validation loss values.
    """

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


def get_loader(
    data_df: pd.DataFrame,
    embedding_field: str,
    label_field: str,
    batch_size: Optional[int] = 32,
    device: Optional[str] = "cuda",
    shuffle: Optional[bool] = True,
) -> torch.utils.data.DataLoader:
    """
    Create a PyTorch DataLoader for a dataset stored in a Pandas DataFrame.

    Args:
        data_df (pd.DataFrame): The Pandas DataFrame containing the dataset.
        embedding_field (str): The name of the DataFrame column containing feature embeddings.
        label_field (str): The name of the DataFrame column containing labels.
        batch_size (Optional[int]): Batch size for data loading. Default is 32.
        device (Optional[str]): The device to move data to (e.g., "cuda" for GPU). Default is "cuda".
        shuffle (Optional[bool]): Whether to shuffle the data. Default is True.

    Returns:
        torch.utils.data.DataLoader: A PyTorch DataLoader for the dataset.
    """
    features = torch.vstack(data_df[embedding_field].tolist()).to(device)
    labels = torch.tensor(data_df[label_field].tolist(), dtype=torch.long).to(device)
    features.requires_grad = False
    labels.requires_grad = False
    assert len(features) == len(labels)
    dataset = torch.utils.data.TensorDataset(features, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )
    return dataloader


def run_eval(
    model: torch.nn.Module,
    eval_dataloader: torch.utils.data.DataLoader,
    verbose: Optional[bool] = False,
) -> Dict[str, float]:
    """
    Run evaluation on a PyTorch model using a given data loader.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        eval_dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        verbose (Optional[bool]): If True, print evaluation metrics, including loss, accuracy,
            F1 score, MCC, and confusion matrix.

    Returns:
        Dict[str, float]: A dictionary containing evaluation metrics, including "Accuracy",
            "F1 Score", and "Mathew correlation coefficient".
    """

    predictions = []
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        avg_loss = 0
        labels = []
        for i, (features, label) in enumerate(eval_dataloader):
            outputs = model(features)
            loss = criterion(outputs, label)
            avg_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            predictions.extend(pred.tolist())
            labels.extend(label.tolist())

    avg_loss /= len(eval_dataloader)
    labels = np.array(labels)
    predictions = np.array(predictions)
    acc = metrics.accuracy_score(labels, predictions)
    f1 = metrics.f1_score(labels, predictions, average="macro")
    cm = metrics.confusion_matrix(labels, predictions)
    mcc = metrics.matthews_corrcoef(labels, predictions)

    stats = {"Accuracy": acc, "F1 Score": f1, "Mathew correleation coefficient": mcc}
    if verbose:
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    return stats
