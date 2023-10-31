import gc
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
import traceback


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from pandarallel import pandarallel
from opensource_paraphrase_generation import (
    get_bucket_data_loader,
    get_save_path,
    save_strings_to_files,
    clear_cache_and_display,
)

pandarallel.initialize(progress_bar=False, nb_workers=2)


def prompt_template_fn(review):
    prompt = f"Review: {review}\nParaphrase of the review:"
    return prompt


def paraphrase(
    model, tokenizer, dataloader, temperature, dataset_file_name, max_new_tokens
):
    print("running for:", temperature, max_new_tokens)
    print(f"running for temperature={temperature}, max_new_tokens={max_new_tokens}")
    paraphrased_list = []
    indices_list = []
    save_path = get_save_path(
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
        dataset_file_name=dataset_file_name,
    )
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"save path is set to : {save_path}")
    for batch in tqdm(dataloader):
        indices, model_inputs = batch
        outputs = model.generate(
            **model_inputs,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            do_sample=True,
            max_new_tokens=max_new_tokens,
        )

        decoded_output = tokenizer.batch_decode(
            outputs[:, model_inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        save_strings_to_files(
            number_list=indices, string_list=decoded_output, root_dir=save_path
        )
        paraphrased_list += decoded_output
        indices_list += indices

        torch.cuda.empty_cache()
        gc.collect()

    model_name = tokenizer.name_or_path
    model_precision = model.parameters().__next__().dtype

    df = pd.DataFrame(
        paraphrased_list,
        index=indices_list,
        columns=[f"{model_name}_{model_precision}_t={temperature}"],
    ).sort_index(ascending=True)
    return df


def get_paraphrases_for_df(
    model,
    tokenizer,
    dataset_file_name,
    max_context_len,
    batch_size,
    iteration,
    temperature_list,
    max_token_generation,
    df_save_prefix,
):
    for temperature in temperature_list:
        dataloader = get_bucket_data_loader(
            dataset_file_name=dataset_file_name,
            model=model,
            tokenizer=tokenizer,
            max_context_len=max_context_len,
            prompt_template_fn=prompt_template_fn,
            temperature=temperature,
            iteration=iteration,
            batch_size=batch_size,
            idxs_for_loader=None,
        )

        try:
            para_df = paraphrase(
                model=model,
                tokenizer=tokenizer,
                dataloader=dataloader,
                temperature=temperature,
                dataset_file_name=dataset_file_name,
                max_new_tokens=max_token_generation,
            )
            para_df.to_csv(
                f"{df_save_prefix}_{max_context_len}_{temperature}_{iteration}_{dataset_file_name}.csv",
                index=False,
            )
        except Exception as e:
            with open("log.txt", "a") as f:
                f.write(str(e))
                f.write(traceback.format_exc())

            print(traceback.format_exc())
            print(sys.exc_info()[2])

        clear_cache_and_display()


def load_model(model_name):
    torch_dtype = "float16"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=getattr(torch, torch_dtype)
    )
    clear_cache_and_display()
    return tokenizer, model


max_context_len = 5000
batch_size = 32
iteration = 1
temperature_list = [0.75, 1.0, 1.25, 1.5, 1.75]
max_token_generation = 150

model_name = ""
tokenizer, model = load_model(model_name)
dataset_file_name = ""


paraphrase_df = get_paraphrases_for_df(
    model=model,
    tokenizer=tokenizer,
    dataset_file_name=dataset_file_name,
    max_context_len=max_context_len,
    batch_size=batch_size,
    iteration=iteration,
    temperature_list=temperature_list,
    max_token_generation=max_token_generation,
    df_save_prefix=model.name,
)
