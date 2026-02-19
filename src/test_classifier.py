import asyncio
import os
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModel, AutoTokenizer

from config import CLF_PATH, HF_SAVE_DIR, LOGS_DIR, MAX_LENGTH

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(HF_SAVE_DIR)
model = AutoModel.from_pretrained(HF_SAVE_DIR).to(device)
model.eval()

artifact = joblib.load(CLF_PATH)
clf = artifact["classifier"]
threshold = 0.65

@torch.no_grad()
def embed_one(text: str) -> np.ndarray:
    enc = tokenizer([text], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    outputs = model(**enc)
    cls_emb = outputs.last_hidden_state[:, 0, :]
    return cls_emb.cpu().numpy()


def is_spam_sync(text: str) -> tuple[bool, float]:
    text = (text or "").strip()
    if not text: return False, 0.0
    emb = embed_one(text)
    proba_spam = float(clf.predict_proba(emb)[0, 1])
    return proba_spam >= threshold, proba_spam


async def is_spam(text: str) -> tuple[bool, float]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, is_spam_sync, text)


def main() -> None:
    path = LOGS_DIR / "messages.csv"
    print(f"[INFO] Loading test data from: {path}")
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if "Message" not in df.columns or "Label" not in df.columns: raise ValueError("messages.csv должен содержать колонки 'Message' и 'Label'")

    label_map = {
        "ham": 0,
        "spam": 1,
        0: 0,
        1: 1,
        "0": 0,
        "1": 1,
    }

    df["y_true"] = df["Label"].map(label_map)
    df = df.dropna(subset=["Message", "y_true"])
    df["y_true"] = df["y_true"].astype(int)

    texts = df["Message"].astype(str).tolist()
    y_true = df["y_true"].values

    print(f"[INFO] Test samples: {len(texts)}")

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

    print("\n=== Confusion matrix (true rows / predicted cols) ===")
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    print(cm)

    print("\n=== Classification report ===")
    print(classification_report(y_true, preds, digits=4))

    acc = accuracy_score(y_true, preds)
    print(f"\nAccuracy: {acc:.4f}")

    df["pred"] = preds
    df["proba_spam"] = probs
    out_path = LOGS_DIR / "test_with_preds.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
