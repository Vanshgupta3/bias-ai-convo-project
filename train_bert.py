"""
Fine-tunes BERT on the bias dataset (train.csv/val.csv/test.csv from split_dataset.py)
and reports accuracy/precision/recall/F1 per class + a confusion matrix on the held-out test set.

Usage:
    python train_bert.py --epochs 4 --batch_size 16
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)


class BiasDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred, id2label):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    report = classification_report(
        labels, preds, target_names=[id2label[i] for i in range(len(id2label))],
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--out_dir", type=str, default="bias_model")
    args = parser.parse_args()

    train_df = pd.read_csv("train.csv", sep="\t")
    val_df = pd.read_csv("val.csv", sep="\t")
    test_df = pd.read_csv("test.csv", sep="\t")

    labels_sorted = sorted(train_df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels_sorted)}
    id2label = {i: l for l, i in label2id.items()}

    train_df["label_id"] = train_df["label"].map(label2id)
    val_df["label_id"] = val_df["label"].map(label2id)
    test_df["label_id"] = test_df["label"].map(label2id)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(labels_sorted),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = BiasDataset(train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer)
    val_ds = BiasDataset(val_df["text"].tolist(), val_df["label_id"].tolist(), tokenizer)
    test_ds = BiasDataset(test_df["text"].tolist(), test_df["label_id"].tolist(), tokenizer)

    training_args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=20,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=lambda ep: compute_metrics(ep, id2label),
    )

    trainer.train()

    # Final evaluation on the held-out TEST set (never seen during training/val selection)
    test_output = trainer.predict(test_ds)
    preds = np.argmax(test_output.predictions, axis=1)
    true = test_df["label_id"].values

    report = classification_report(
        true, preds, target_names=labels_sorted, zero_division=0
    )
    cm = confusion_matrix(true, preds)

    print("\n=== TEST SET RESULTS ===")
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print(labels_sorted)
    print(cm)

    with open("test_report.json", "w") as f:
        json.dump(
            classification_report(true, preds, target_names=labels_sorted, output_dict=True, zero_division=0),
            f, indent=2,
        )

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"\nModel saved to {args.out_dir}/")
    print("Full test report saved to test_report.json")


if __name__ == "__main__":
    main()