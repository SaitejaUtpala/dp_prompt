import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn import metrics
import pandas as pd 
from typing import Dict, Optional

Stats = Dict[str, Dict[str, float]]
static_trainers = {}
    

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
max_length = 512


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = metrics.accuracy_score(labels, predictions)
    f1 = metrics.f1_score(labels, predictions, average="macro")
    mcc = metrics.matthews_corrcoef(labels, predictions)

    stats = {"Accuracy": acc, "F1 Score": f1, "Mathew correleation coefficient": mcc}
    return stats


    
def get_trained_trainer(train_dataset, val_dataset, num_labels, label):
    print("num labels", num_labels)
    if label in static_trainers:
        return static_trainers[label]

    training_args = TrainingArguments(
        output_dir=f"./output_{label}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        logging_steps=300,
        load_best_model_at_end=True,
        metric_for_best_model="F1 Score",
    )

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=num_labels
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    static_trainers[label] = trainer
    return trainer


def run_eval(trainer, test_dataset):
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    return test_results



def get_stats(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_text_field: str,
    val_text_field: str,
    test_text_field: str,
    sentiment_id_field: Optional[str] = "sentiment_id",
    author_id_field: Optional[str] = "author_id",
    verbose: Optional[bool] = False,
) -> Stats:
    labels_fields = [sentiment_id_field, author_id_field]
    label_field_stats = {}
    tokenized_train_data = tokenizer(
        train_df[train_text_field].tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    tokenized_val_data = tokenizer(
        val_df[val_text_field].tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    tokenized_test_data = tokenizer(
        test_df[test_text_field].tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    for label_field in labels_fields:
        num_classes = train_df[label_field].nunique()
        train_labels = np.array(train_df[label_field])
        val_labels = np.array(val_df[label_field])
        test_labels = np.array(test_df[label_field])

        train_dataset = Dataset.from_dict(
            {
                "input_ids": tokenized_train_data["input_ids"],
                "attention_mask": tokenized_train_data["attention_mask"],
                "labels": train_labels,
            }
        )

        val_dataset = Dataset.from_dict(
            {
                "input_ids": tokenized_val_data["input_ids"],
                "attention_mask": tokenized_val_data["attention_mask"],
                "labels": val_labels,
            }
        )

        test_dataset = Dataset.from_dict(
            {
                "input_ids": tokenized_test_data["input_ids"],
                "attention_mask": tokenized_test_data["attention_mask"],
                "labels": test_labels,
            }
        )

        trainer = get_trained_trainer(train_dataset, val_dataset, num_labels=num_classes, label = label_field)
        test_stats = run_eval(trainer, test_dataset)
        print("----------test stats-------")
        print(test_stats)
        label_field_stats[label_field] = test_stats

    return label_field_stats