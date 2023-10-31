import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Optional, Dict, List, Tuple
from sklearn.model_selection import train_test_split


import json
import pandas as pd
import numpy as np
from sklearn import metrics
from pathlib import Path
from typing import Optional, List, Dict


Stats = Dict[str, Dict[str, float]]


def attack(
    clean_df: pd.DataFrame,
    private_df: pd.DataFrame,
    clean_df_text_field: str,
    private_df_text_field: str,
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

        train_text_field = clean_df_text_field
        val_text_field = clean_df_text_field
        test_text_field = private_df_text_field
    
    elif attack_type == "adaptive":
        train_df = private_df[private_df.index.isin(train_idx_list)]
        val_df = private_df[private_df.index.isin(val_idx_list)]
        test_df = private_df[private_df.index.isin(test_idx_list)]

        train_text_field = private_df_text_field
        val_text_field = private_df_text_field
        test_text_field = private_df_text_field
    else:
        print("Error this shouldn't happend!")

    print(train_df.shape)
    print(val_df.shape)
    print(test_df.shape)
    print(train_text_field)
    print(sentiment_id_field)
    print(author_id_field)
    print(val_text_field)
    print(test_text_field)
    train_df = train_df[[train_text_field, sentiment_id_field, author_id_field]].dropna()
    val_df = val_df[[val_text_field, sentiment_id_field, author_id_field]].dropna()
    test_df = test_df[[test_text_field, sentiment_id_field, author_id_field]].dropna()
    
    print(train_df.shape)
    print(val_df.shape)
    print(test_df.shape)
    


    stats = get_stats(
        train_df=train_df,
        test_df=test_df,
        val_df=val_df,
        train_text_field=train_text_field,
        val_text_field=val_text_field,
        test_text_field=test_text_field,
        sentiment_id_field=sentiment_id_field,
        author_id_field=author_id_field,
        verbose=verbose
    )
    if save:
        assert save_path is not None
        with open(save_path, "w") as fp:
            json.dump(stats, fp)

    return stats


def get_text_attack_stats(
    clean_df, private_df, columns_to_run, train_idx_list, val_idx_list, test_idx_list
):
    assert all(
        column in clean_df.columns for column in ["review", "sentiment_id", "author_id"]
    )
    assert all(
        column not in private_df.columns for column in ["sentiment_id", "author_id"]
    )

    attack_types = ["static"]
    
    
    model_stats = {}
    for private_column in columns_to_run:
        print(f"doing it for {private_column}")
        model_attack_stats = {}
        for attack_type in attack_types: 
            attack_stats = attack(
                clean_df=clean_df,
                private_df=private_df,
                clean_df_text_field="review",
                private_df_text_field=private_column,
                train_idx_list=train_idx_list,
                val_idx_list=val_idx_list,
                test_idx_list=test_idx_list,
                attack_type=attack_type,
                verbose=False
            )
            model_attack_stats[attack_type] = attack_stats
        model_stats[private_column] = model_attack_stats
        
        with open(f"statictext_imdb_custext.json", "w") as fp:
            json.dump(model_stats, fp)

    return model_stats