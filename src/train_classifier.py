import os
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

from config import HF_SAVE_DIR, CLF_PATH, MAX_LENGTH, MODEL_NAME, MISSES_PATH, CSV_PATH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading HF model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


@torch.no_grad()
def encode_texts(texts, batch_size: int = 32) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


print("Loading data...")
df = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip")

label_map = {
    "ham": 0,
    "spam": 1,
    0: 0,
    1: 1,
    "0": 0,
    "1": 1,
}
df["y"] = df["Label"].map(label_map)

df = df.dropna(subset=["Message", "y"])
df["y"] = df["y"].astype(int)

texts = df["Message"].astype(str).tolist()
y = df["y"].values

print(f"Total samples: {len(texts)}, spam: {sum(y)}, ham: {len(y) - sum(y)}")

X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Encoding train texts...")
X_train_emb = encode_texts(X_train_texts)

print("Encoding test texts...")
X_test_emb = encode_texts(X_test_texts)

print("Training classifier...")
clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
clf.fit(X_train_emb, y_train)

print("\n=== Evaluation on test split ===")
y_pred = clf.predict(X_test_emb)
y_proba = clf.predict_proba(X_test_emb)[:, 1]

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

ham_prec, spam_prec = prec
ham_rec, spam_rec = rec
ham_f1, spam_f1 = f1
ham_sup, spam_sup = support

print("Accuracy: {:.4f}".format(acc))
print("\nPer-class metrics:")
print("  HAM  (0): prec={:.4f}, rec={:.4f}, f1={:.4f}, support={}".format(ham_prec, ham_rec, ham_f1, ham_sup))
print("  SPAM (1): prec={:.4f}, rec={:.4f}, f1={:.4f}, support={}".format(spam_prec, spam_rec, spam_f1, spam_sup))


tn, fp, fn, tp = cm.ravel()
print("\nConfusion matrix [labels: 0=HAM, 1=SPAM]:")
print(cm)
print(f"  TN (ham→ham): {tn}")
print(f"  FP (ham→spam): {fp}")
print(f"  FN (spam→ham): {fn}")
print(f"  TP (spam→spam): {tp}")

miss_records = []
for text, true_label, pred_label, proba in zip(X_test_texts, y_test, y_pred, y_proba):
    if pred_label != true_label:
        miss_type = "FN" if true_label == 1 else "FP"
        miss_records.append({
            "Message": text,
            "TrueLabel": int(true_label),
            "PredLabel": int(pred_label),
            "Proba": float(proba),
            "MissType": miss_type,
        })

pd.DataFrame(miss_records).to_csv(MISSES_PATH, index=False, encoding="utf-8")
print(f"Saved misses to {MISSES_PATH}")

print("\nSaving HF model & tokenizer...")
tokenizer.save_pretrained(HF_SAVE_DIR)
model.save_pretrained(HF_SAVE_DIR)

print("Saving classifier with joblib...")
joblib.dump(
    {
        "classifier": clf,
        "label_map": label_map,
        "threshold": 0.5,
    },
    CLF_PATH,
)

print("Done.")
