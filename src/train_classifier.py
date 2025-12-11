import os
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,                  # >>>
    precision_recall_fscore_support, # >>>
    confusion_matrix                 # >>>
)
from sklearn.model_selection import train_test_split
import joblib

from config import HF_SAVE_DIR, CLF_PATH, MAX_LENGTH, MODEL_NAME, MISSES_PATH, CSV_PATH

# good starting point

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. Load HF model
print("Loading HF model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


# 3. Function to embed texts with the transformer
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
            return_tensors="pt"
        )

        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)

        # Use [CLS] token representation (first token)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


# 4. Load your data
print("Loading data...")
df = pd.read_csv(CSV_PATH)

# Map labels to 0/1
label_map = {
    "ham": 0,
    "spam": 1,
    0: 0,
    1: 1
}
df["y"] = df["Label"].map(label_map)

# Drop any rows that didn't map correctly
df = df.dropna(subset=["y"])
df["y"] = df["y"].astype(int)

texts = df["Message"].astype(str).tolist()
y = df["y"].values

print(f"Total samples: {len(texts)}, spam: {sum(y)}, ham: {len(y) - sum(y)}")

# 5. Train/test split
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Embed texts with transformer
print("Encoding train texts...")
X_train_emb = encode_texts(X_train_texts)

print("Encoding test texts...")
X_test_emb = encode_texts(X_test_texts)

# 7. Train simple classifier (Logistic Regression)
print("Training classifier...")
clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",   # spam/ham imbalance
    n_jobs=-1
)
clf.fit(X_train_emb, y_train)

# 8. Evaluate
print("\n=== Evaluation on test split ===")
y_pred = clf.predict(X_test_emb)
y_proba = clf.predict_proba(X_test_emb)[:, 1]

# Стандартный отчёт
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

# >>> Дополнительные метрики
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=[0, 1]
)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

ham_prec, spam_prec = prec
ham_rec, spam_rec = rec
ham_f1, spam_f1 = f1
ham_sup, spam_sup = support

print("Accuracy: {:.4f}".format(acc))
print("\nPer-class metrics:")
print("  HAM  (0): prec={:.4f}, rec={:.4f}, f1={:.4f}, support={}".format(
    ham_prec, ham_rec, ham_f1, ham_sup
))
print("  SPAM (1): prec={:.4f}, rec={:.4f}, f1={:.4f}, support={}".format(
    spam_prec, spam_rec, spam_f1, spam_sup
))

tn, fp, fn, tp = cm.ravel()
print("\nConfusion matrix [labels: 0=HAM, 1=SPAM]:")
print(cm)
print(f"  TN (ham→ham): {tn}")
print(f"  FP (ham→spam): {fp}  <-- лишние баны")
print(f"  FN (spam→ham): {fn}  <-- пропущенный спам")
print(f"  TP (spam→spam): {tp}")

# Небольшая подсказка по качеству
print("\nHeuristic hints:")
if spam_rec < 0.9:
    print(f"  • spam_recall={spam_rec:.3f} < 0.90 → модель ПРОПУСКАЕТ часть спама, стоит добирать примеры спама")
else:
    print(f"  • spam_recall={spam_rec:.3f} ≥ 0.90 → пропусков спама немного (по этому сплиту)")

if spam_prec < 0.9:
    print(f"  • spam_precision={spam_prec:.3f} < 0.90 → много ложных банов, можно поднять threshold или добрать нормальных сообщений")
else:
    print(f"  • spam_precision={spam_prec:.3f} ≥ 0.90 → с ложными банами всё неплохо")

print(f"  • общая accuracy={acc:.3f} (ориентир, но важнее метрики по классу SPAM)")

# 8.1. Log misses (FP & FN)
miss_records = []
for text, true_label, pred_label, proba in zip(X_test_texts, y_test, y_pred, y_proba):
    if pred_label != true_label:
        if true_label == 1 and pred_label == 0:
            miss_type = "FN"  # false negative: spam пропущен
        elif true_label == 0 and pred_label == 1:
            miss_type = "FP"  # false positive: лишний бан
        else:
            miss_type = "OTHER"

        miss_records.append(
            {
                "Message": text,
                "TrueLabel": int(true_label),
                "PredLabel": int(pred_label),
                "Proba": float(proba),
                "MissType": miss_type,
            }
        )

if miss_records:
    misses_df = pd.DataFrame(miss_records)

    # если файл уже есть — аккуратно добавляем
    if os.path.exists(MISSES_PATH):
        old = pd.read_csv(MISSES_PATH)
        misses_df = pd.concat([old, misses_df], ignore_index=True)

    misses_df.to_csv(MISSES_PATH, index=False, encoding="utf-8")
    print(f"\nLogged {len(miss_records)} misses to {MISSES_PATH}")
else:
    print("\nNo misses on test set (perfect split).")

# 9. Save artifacts
print("\nSaving HF model & tokenizer...")
tokenizer.save_pretrained(HF_SAVE_DIR)
model.save_pretrained(HF_SAVE_DIR)

print("Saving classifier with joblib...")
joblib.dump(
    {
        "classifier": clf,
        "label_map": label_map,
        "threshold": 0.5  # you can tune this later
    },
    CLF_PATH
)

print("Done.")