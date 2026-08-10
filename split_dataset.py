"""
Stratified train/val/test split for the bias dataset.

Usage:
    python split_dataset.py --in bias_dataset.csv
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", type=str, default="bias_dataset.csv")
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)
    args = parser.parse_args()

    df = pd.read_csv(args.infile, sep="\t")
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset="text").reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df, train_size=args.train_frac, stratify=df["label"], random_state=42
    )
    val_size = args.val_frac / (1 - args.train_frac)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_size, stratify=temp_df["label"], random_state=42
    )

    train_df.to_csv("train.csv", sep="\t", index=False)
    val_df.to_csv("val.csv", sep="\t", index=False)
    test_df.to_csv("test.csv", sep="\t", index=False)

    print(f"Total (deduped): {len(df)}")
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print("\nTrain class balance:")
    print(train_df["label"].value_counts())


if __name__ == "__main__":
    main()