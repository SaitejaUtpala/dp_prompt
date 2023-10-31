"""Utility files for saving and processing result files."""

import re
import os
import gc
import torch
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from typing import List, Callable, Optional
from transformers import AutoModel, AutoTokenizer



def clear_cache_and_display():
    
    torch.cuda.empty_cache()
    gc.collect()

    sp = subprocess.Popen(
        ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8")
    print(out_list)
    
    
    
def zip_dir(save_path: Path, dir_name: Path):
    shutil.make_archive(save_path, "zip", dir_name)


def save_strings_to_files(
    number_list: List[int], string_list: List[int], root_dir: Path
) -> List:
    filenames = []
    for idx, content in zip(number_list, string_list):
        n = 1
        filename = f"idx={str(idx)}_n={n}.txt"
        while os.path.exists(os.path.join(root_dir, filename)):
            filename = f"idx={str(idx)}_n={n}.txt"
            n += 1
        filepath = os.path.join(root_dir, filename)
        with open(filepath, "w", encoding="UTF-8") as file:
            file.write(content)
        filenames.append(filepath)
    return filenames


def get_save_path(
    model: AutoModel, tokenizer: AutoTokenizer, temperature: int, dataset_file_name: str
) -> Path:
    dataset_name = dataset_file_name.split("_")[1]
    model_name = tokenizer.name_or_path
    model_precision = model.parameters().__next__().dtype
    if "/" in model_name:
        model_name = model_name.split("/")[1]
    save_path = Path(f"{model_name}/{model_precision}/{temperature}/{dataset_name}")
    return save_path


def build_idx_to_n_dict(root_dir: Path, n_find: int, N: int) -> List[int]:
    file_list = os.listdir(root_dir)

    idx_n_dict = {}

    for file_name in file_list:
        match = re.match(r"idx=(\d+)_n=(\d+).txt", file_name)
        if match:
            idx = int(match.group(1))
            n = int(match.group(2))
            if idx not in idx_n_dict:
                idx_n_dict[idx] = []
            idx_n_dict[idx].append(n)

    index_universe = np.arange(N)
    indexes_without_n = []
    for idx in index_universe:
        if idx not in idx_n_dict:
            indexes_without_n.append(idx)
        elif n_find not in idx_n_dict[idx]:
            indexes_without_n.append(idx)

    return indexes_without_n


def left_over_idx(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    temperature: int,
    dataset_file_name: str,
    n_find: int,
) -> List[int]:
    data_df = pd.read_csv(dataset_file_name)
    save_path = get_save_path(
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
        dataset_file_name=dataset_file_name,
    )
    save_path.mkdir(parents=True, exist_ok=True)
    left = build_idx_to_n_dict(root_dir=save_path, n_find=n_find, N=len(data_df))
    return left


def bucket_batch_sampler(
    prompt_text_list: List[str],
    indices: List[int],
    max_context_len: int,
    batch_size: 8,
    bucket_size_multiplier: 100,
):
    """Bucket Iterator used to group sequeces of strings similar
    length, so that amount of padding is reduced.
    """
    # filter the indices
    print("before", len(indices))
    indices = [x for x in indices if x[1] <= max_context_len]
    print("after", len(indices))

    pooled_indices = []
    bucket_size = batch_size * bucket_size_multiplier
    for i in range(0, len(indices), bucket_size):
        pooled_indices.extend(sorted(indices[i : i + bucket_size], key=lambda x: x[1]))
    pooled_indices = [x[0] for x in pooled_indices]

    # yield indices for current batch
    for i in range(0, len(pooled_indices), batch_size):
        yield pooled_indices[i : i + batch_size]


def collate_batch(batch, tokenizer, device="cuda"):
    indices_list, text_list = map(list, zip(*batch))
    return (
        indices_list,
        tokenizer(text_list, padding="longest", return_tensors="pt").to(device),
    )


def get_bucket_data_loader(
    dataset_file_name: str,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    max_context_len: int,
    prompt_template_fn: Callable,
    temperature: int,
    iteration: int,
    idxs_for_loader: Optional[List[int]] = None,
    batch_size: Optional[int] = 8,
    bucket_size_multiplier: Optional[int] = 1000,
):
    if idxs_for_loader is None:
        idxs_for_loader = left_over_idx(
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            dataset_file_name=dataset_file_name,
            n_find=iteration,
        )
        print("idxs_for_loader is None, finding left_over_idx.")

    full_data_df = pd.read_csv(dataset_file_name)
    full_data_df["prompt"] = full_data_df["review"].apply(lambda row: prompt_template_fn(row))
    full_data_df["prompt_len"] = full_data_df["prompt"].parallel_apply(
        lambda x: len(tokenizer.encode(x))
    )
    prompt_text_list = full_data_df["prompt"].tolist()
    full_data = [x for x in full_data_df[["prompt"]].itertuples(index=True)]
    
    
    dataloader_df = full_data_df[full_data_df.index.isin(idxs_for_loader)]
    indices = [tuple(x) for x in dataloader_df[["prompt_len"]].itertuples(index=True)]
    print(f"Built dataloader for dataset of size: {len(dataloader_df)}")

    batch_dataloader = DataLoader(
        full_data,
        batch_sampler=bucket_batch_sampler(
            prompt_text_list=prompt_text_list,
            indices=indices,
            max_context_len=max_context_len,
            batch_size=batch_size,
            bucket_size_multiplier=bucket_size_multiplier,
        ),
        collate_fn=lambda x: collate_batch(x, tokenizer),
        shuffle=False,
        drop_last=False,
    )
    return batch_dataloader
