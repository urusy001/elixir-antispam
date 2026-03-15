import asyncio
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModel, AutoTokenizer

from config import CLF_PATH, HF_SAVE_DIR, LOGS_DIR, MAX_LENGTH

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_message(text: str) -> str:
    return " ".join(str(text).strip().split())


print("Loading tokenizer/model...")
tokenizer = AutoTokenizer.from_pretrained(str(HF_SAVE_DIR), local_files_only=True)
model = AutoModel.from_pretrained(str(HF_SAVE_DIR), local_files_only=True).to(device)
model.eval()

print("Loading classifier artifact...")
artifact = joblib.load(str(CLF_PATH))
clf = artifact["classifier"]

# Uses trained threshold by default
threshold = float(artifact.get("threshold", 0.5)) #97.72

# Optional manual override:
threshold = 0.56 #97.99

print(f"Using threshold: {threshold:.4f}")


@torch.no_grad()
def embed_one(text: str) -> np.ndarray:
    enc = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    outputs = model(**enc)
    cls_emb = outputs.last_hidden_state[:, 0, :]
    return cls_emb.cpu().numpy()


def is_spam_sync(text: str) -> tuple[bool, float]:
    text = normalize_message(text)
    if not text:
        return False, 0.0

    emb = embed_one(text)
    proba_spam = float(clf.predict_proba(emb)[0, 1])
    return proba_spam >= threshold, proba_spam


async def is_spam(text: str) -> tuple[bool, float]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, is_spam_sync, text)


def print_metrics(y_true: np.ndarray, preds: np.ndarray) -> None:
    print("\n=== Confusion matrix (true rows / predicted cols) ===")
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    print("\n=== Classification report ===")
    print(classification_report(y_true, preds, labels=[0, 1], digits=4, zero_division=0))

    acc = accuracy_score(y_true, preds)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true,
        preds,
        labels=[0, 1],
        zero_division=0,
    )

    ham_prec, spam_prec = prec
    ham_rec, spam_rec = rec
    ham_f1, spam_f1 = f1
    ham_sup, spam_sup = support

    print(f"Accuracy: {acc:.4f}")
    print("\nPer-class metrics:")
    print(
        "  HAM  (0): prec={:.4f}, rec={:.4f}, f1={:.4f}, support={}".format(
            ham_prec, ham_rec, ham_f1, ham_sup
        )
    )
    print(
        "  SPAM (1): prec={:.4f}, rec={:.4f}, f1={:.4f}, support={}".format(
            spam_prec, spam_rec, spam_f1, spam_sup
        )
    )

    print("\nDetailed counts:")
    print(f"  TN (ham→ham):  {tn}")
    print(f"  FP (ham→spam): {fp}")
    print(f"  FN (spam→ham): {fn}")
    print(f"  TP (spam→spam): {tp}")


def main() -> None:
    path = Path(LOGS_DIR) / "messages.csv"
    print(f"[INFO] Loading test data from: {path}")

    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if "Message" not in df.columns or "Label" not in df.columns:
        raise ValueError("messages.csv must contain columns 'Message' and 'Label'")

    label_map = {
        "ham": 0,
        "spam": 1,
        0: 0,
        1: 1,
        "0": 0,
        "1": 1,
    }

    df["y_true"] = df["Label"].map(label_map)
    df = df.dropna(subset=["Message", "y_true"]).copy()
    df["Message"] = df["Message"].astype(str).map(normalize_message)
    df = df[df["Message"] != ""].copy()
    df["y_true"] = df["y_true"].astype(int)

    texts = df["Message"].tolist()
    y_true = df["y_true"].values

    print(f"[INFO] Test samples after cleaning: {len(texts)}")

    preds = []
    probs = []

    for i, text in enumerate(texts, 1):
        is_spam_flag, proba_spam = is_spam_sync(text)
        preds.append(int(is_spam_flag))
        probs.append(proba_spam)

        if i % 100 == 0 or i == len(texts):
            print(f"[INFO] Processed {i}/{len(texts)}")

    preds = np.array(preds, dtype=int)
    probs = np.array(probs, dtype=float)

    print_metrics(y_true, preds)

    df["pred"] = preds
    df["pred_label"] = np.where(df["pred"] == 1, "spam", "ham")
    df["proba_spam"] = probs
    df["threshold_used"] = threshold
    df["is_correct"] = (df["pred"] == df["y_true"]).astype(int)

    def get_error_type(row):
        if row["pred"] == row["y_true"]:
            return ""
        if row["y_true"] == 0 and row["pred"] == 1:
            return "FP"
        if row["y_true"] == 1 and row["pred"] == 0:
            return "FN"
        return ""

    df["error_type"] = df.apply(get_error_type, axis=1)

    out_dir = Path(LOGS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "test_with_preds.csv"
    fp_path = out_dir / "false_positives.csv"
    fn_path = out_dir / "false_negatives.csv"

    df.to_csv(out_path, index=False, encoding="utf-8")
    df[df["error_type"] == "FP"].to_csv(fp_path, index=False, encoding="utf-8")
    df[df["error_type"] == "FN"].to_csv(fn_path, index=False, encoding="utf-8")

    print(f"\n[INFO] Saved predictions to {out_path}")
    print(f"[INFO] Saved false positives to {fp_path}")
    print(f"[INFO] Saved false negatives to {fn_path}")

    # Show most confident mistakes
    mistakes = df[df["error_type"] != ""].copy()

    if not mistakes.empty:
        print("\n=== Top 10 highest-confidence mistakes ===")
        mistakes["confidence"] = np.where(
            mistakes["pred"] == 1,
            mistakes["proba_spam"],
            1.0 - mistakes["proba_spam"],
        )
        mistakes = mistakes.sort_values("confidence", ascending=False)

        for _, row in mistakes.head(10).iterrows():
            msg = row["Message"]
            if len(msg) > 180:
                msg = msg[:180] + "..."
            print(
                f"[{row['error_type']}] true={row['y_true']} pred={row['pred']} "
                f"proba_spam={row['proba_spam']:.4f} | {msg}"
            )
    else:
        print("\nNo mistakes found.")


if __name__ == "__main__":
    main()