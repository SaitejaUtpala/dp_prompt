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
CLEANR = re.compile("<.*?>")



def split_into_sentences(doc: str) -> List[str]:
    try :
        return sent_tokenize(doc)
    except Exception as e:
        print("yikes")
        print(type(doc))
        print(doc)
        return [""]


def preprocess_df(data_df: pd.DataFrame, field: str) -> pd.Series:
    return data_df[field].parallel_apply(split_into_sentences)


def encode(
    data_df: pd.DataFrame,
    model: torch.nn.Module,
    encode_field: str,
    device: Optional[str] = "cuda",
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
    train_epochs: Optional[int] = 100,
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
    train_epochs: Optional[int] = 20,
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


def get_loader(
    data_df: pd.DataFrame,
    embedding_field: str,
    label_field: str,
    batch_size: Optional[int] = 32,
    device: Optional[str] = "cuda",
    shuffle: Optional[bool] = True,
) -> torch.utils.data.DataLoader:
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
    model.eval()

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