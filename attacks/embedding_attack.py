import re
import pandas as pd
from typing import List, Optional, Dict, List, Tuple
from pandarallel import pandarallel
from sklearn.preprocessing import LabelEncoder
from attacks.embedding_attacker_architecture import encode_df, get_loader, get_train_mlp, run_eval

pandarallel.initialize(progress_bar=True)
CLEANR = re.compile("<.*?>")


import json
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, TypeAlias


Stats: TypeAlias = Dict[str, Dict[str, float]]


def get_stats(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_embedding_field: str,
    val_embedding_field: str,
    test_embedding_field: str,
    sentiment_id_field: Optional[str] = "sentiment_id",
    author_id_field: Optional[str] = "author_id",
    verbose: Optional[bool] = False
) -> Stats:
    labels_fields = [sentiment_id_field, author_id_field]

    label_field_stats = {}
    for label_field in labels_fields:
        num_classes = train_df[label_field].nunique()
        train_dataloader = get_loader(
            data_df=train_df,
            embedding_field=train_embedding_field,
            label_field=label_field,
        )
        val_dataloader = get_loader(
            data_df=val_df,
            embedding_field=val_embedding_field,
            label_field=label_field,
        )
        test_dataloader = get_loader(
            data_df=test_df,
            embedding_field=test_embedding_field,
            label_field=label_field,
        )

        trained_model, train_losses, val_acc = get_train_mlp(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            train_epochs=50,
            num_classes=num_classes,
            plot=verbose,
            verbose=verbose
        )
        test_stats = run_eval(trained_model, test_dataloader)
        train_baseline_f1 = run_eval(trained_model, train_dataloader)
        val_baseline_f1 = run_eval(trained_model, val_dataloader)
        label_field_stats[label_field] = test_stats

    return label_field_stats


def attack(
    clean_df: pd.DataFrame,
    private_df: pd.DataFrame,
    clean_df_embedding_field: str,
    private_df_embedding_field: str,
    train_idx_list: List[int],
    val_idx_list: List[int],
    test_idx_list: List[int],
    attack_type: str,
    sentiment_id_field: Optional[str] = "sentiment_id",
    author_id_field: Optional[str] = "author_id",
    save: Optional[bool] = False,
    save_path: Optional[Path] = None,
    verbose: Optional[bool] = False
) -> Stats:
    assert attack_type in ["static", "adaptive"]

    private_df = pd.concat([private_df, clean_df[[sentiment_id_field, author_id_field]]], axis=1)
    if attack_type == "static":
        train_df = clean_df[clean_df.index.isin(train_idx_list)]
        val_df = clean_df[clean_df.index.isin(val_idx_list)]
        test_df = private_df[private_df.index.isin(test_idx_list)]

        train_embedding_field = clean_df_embedding_field
        val_embedding_field = clean_df_embedding_field
        test_embedding_field = private_df_embedding_field
    
    elif attack_type == "adaptive":
        train_df = private_df[private_df.index.isin(train_idx_list)]
        val_df = private_df[private_df.index.isin(val_idx_list)]
        test_df = private_df[private_df.index.isin(test_idx_list)]

        train_embedding_field = private_df_embedding_field
        val_embedding_field = private_df_embedding_field
        test_embedding_field = private_df_embedding_field
    else:
        print("Error this shouldn't happend!")

    train_df = train_df[[train_embedding_field, sentiment_id_field, author_id_field]]
    val_df = val_df[[val_embedding_field, sentiment_id_field, author_id_field]]
    test_df = test_df[[test_embedding_field, sentiment_id_field, author_id_field]]

    stats = get_stats(
        train_df=train_df,
        test_df=test_df,
        val_df=val_df,
        train_embedding_field=train_embedding_field,
        val_embedding_field=val_embedding_field,
        test_embedding_field=test_embedding_field,
        sentiment_id_field=sentiment_id_field,
        author_id_field=author_id_field,
        verbose=verbose
    )
    if save:
        assert save_path is not None
        with open(save_path, "w") as fp:
            json.dump(stats, fp)

    return stats

def get_embedding_attack_stats(
    clean_df, private_df, columns_to_run, train_idx_list, val_idx_list, test_idx_list, encode=True
):
    assert all(
        column in clean_df.columns for column in ["review", "sentiment_id", "author_id"]
    )
    assert all(
        column not in private_df.columns for column in ["sentiment_id", "author_id"]
    )

    attack_types = ["static", "adaptive"]
    if encode:
        encode_df(data_df=private_df, encode_fields=columns_to_run)
        encode_df(data_df=clean_df, encode_fields=["review"])
        columns_to_run = [ f"{col}_embeddings" for col in columns_to_run ]
        
    model_stats = {}
    for private_column_embeddings in columns_to_run:
        print(f"doing it for {private_column_embeddings}")
        model_attack_stats = {}
        for attack_type in attack_types:
            attack_stats = attack(
                clean_df=clean_df,
                private_df=private_df,
                clean_df_embedding_field="review_embeddings",
                private_df_embedding_field=private_column_embeddings,
                train_idx_list=train_idx_list,
                val_idx_list=val_idx_list,
                test_idx_list=test_idx_list,
                attack_type=attack_type,
                verbose=False
            )
            model_attack_stats[attack_type] = attack_stats
        model_stats[private_column_embeddings] = model_attack_stats

    return model_stats