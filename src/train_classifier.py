import json
import joblib
import numpy as np
import pandas as pd
import torch

from pathlib import Path
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

        # CLS embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def normalize_message(text: str) -> str:
    return " ".join(str(text).strip().split())


def evaluate_predictions(y_true, y_pred, title: str) -> dict:
    print(f"\n=== {title} ===")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    ham_prec, spam_prec = prec
    ham_rec, spam_rec = rec
    ham_f1, spam_f1 = f1
    ham_sup, spam_sup = support

    tn, fp, fn, tp = cm.ravel()

    print("Accuracy: {:.4f}".format(acc))
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

    print("\nConfusion matrix [labels: 0=HAM, 1=SPAM]:")
    print(cm)
    print(f"  TN (ham→ham): {tn}")
    print(f"  FP (ham→spam): {fp}")
    print(f"  FN (spam→ham): {fn}")
    print(f"  TP (spam→spam): {tp}")

    return {
        "accuracy": float(acc),
        "ham_precision": float(ham_prec),
        "ham_recall": float(ham_rec),
        "ham_f1": float(ham_f1),
        "ham_support": int(ham_sup),
        "spam_precision": float(spam_prec),
        "spam_recall": float(spam_rec),
        "spam_f1": float(spam_f1),
        "spam_support": int(spam_sup),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_threshold: float = 0.10,
    max_threshold: float = 0.90,
    step: float = 0.01,
) -> tuple[float, dict]:
    """
    Picks threshold by maximizing spam F1 on validation set.
    Tie-breakers:
    1) lower false positives
    2) higher accuracy
    """
    thresholds = np.arange(min_threshold, max_threshold + 1e-9, step)

    best_threshold = 0.5
    best_info = None

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_true, y_pred)

        info = {
            "threshold": float(thr),
            "accuracy": float(acc),
            "ham_precision": float(prec[0]),
            "ham_recall": float(rec[0]),
            "ham_f1": float(f1[0]),
            "spam_precision": float(prec[1]),
            "spam_recall": float(rec[1]),
            "spam_f1": float(f1[1]),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

        if best_info is None:
            best_threshold = float(thr)
            best_info = info
            continue

        better = False

        # Primary objective: spam F1
        if info["spam_f1"] > best_info["spam_f1"]:
            better = True
        elif info["spam_f1"] == best_info["spam_f1"]:
            # Tie-breaker 1: fewer false positives
            if info["fp"] < best_info["fp"]:
                better = True
            elif info["fp"] == best_info["fp"]:
                # Tie-breaker 2: higher accuracy
                if info["accuracy"] > best_info["accuracy"]:
                    better = True

        if better:
            best_threshold = float(thr)
            best_info = info

    return best_threshold, best_info


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

df = df.dropna(subset=["Message", "y"]).copy()
df["Message"] = df["Message"].astype(str).map(normalize_message)
df["y"] = df["y"].astype(int)

# Remove empty messages after normalization
df = df[df["Message"] != ""].copy()

# Remove exact duplicates by normalized message; keep first label seen
before_dedup = len(df)
df = df.drop_duplicates(subset=["Message"]).copy()
after_dedup = len(df)

texts = df["Message"].tolist()
y = df["y"].values

print(f"Removed duplicates: {before_dedup - after_dedup}")
print(f"Total unique samples: {len(texts)}, spam: {sum(y)}, ham: {len(y) - sum(y)}")

# 80/20 split first, then split the 80 into 64/16 train/val
X_train_texts, X_temp_texts, y_train, y_temp = train_test_split(
    texts,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

X_val_texts, X_test_texts, y_val, y_test = train_test_split(
    X_temp_texts,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp,
)

print(f"Train size: {len(X_train_texts)}")
print(f"Val size:   {len(X_val_texts)}")
print(f"Test size:  {len(X_test_texts)}")

print("Encoding train texts...")
X_train_emb = encode_texts(X_train_texts)

print("Encoding val texts...")
X_val_emb = encode_texts(X_val_texts)

print("Encoding test texts...")
X_test_emb = encode_texts(X_test_texts)

print("Training classifier...")
clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)
clf.fit(X_train_emb, y_train)

# Validation probabilities for threshold tuning
y_val_proba = clf.predict_proba(X_val_emb)[:, 1]

print("\nSearching best threshold on validation split...")
best_threshold, best_val_info = choose_best_threshold(y_val, y_val_proba)

print(f"Chosen threshold: {best_threshold:.2f}")
print("Validation metrics at chosen threshold:")
for k, v in best_val_info.items():
    print(f"  {k}: {v}")

# Compare test performance at 0.5 and chosen threshold
y_test_proba = clf.predict_proba(X_test_emb)[:, 1]

y_test_pred_05 = (y_test_proba >= 0.50).astype(int)
metrics_05 = evaluate_predictions(y_test, y_test_pred_05, "Evaluation on TEST split @ threshold=0.50")

y_test_pred_best = (y_test_proba >= best_threshold).astype(int)
metrics_best = evaluate_predictions(y_test, y_test_pred_best, f"Evaluation on TEST split @ chosen threshold={best_threshold:.2f}")

# Save misses using chosen threshold
miss_records = []
for text, true_label, proba in zip(X_test_texts, y_test, y_test_proba):
    pred_label = int(proba >= best_threshold)
    if pred_label != true_label:
        miss_type = "FN" if true_label == 1 else "FP"
        miss_records.append(
            {
                "Message": text,
                "TrueLabel": int(true_label),
                "PredLabel": int(pred_label),
                "Proba": float(proba),
                "Threshold": float(best_threshold),
                "MissType": miss_type,
            }
        )

misses_path = MISSES_PATH
pd.DataFrame(miss_records).to_csv(misses_path, index=False, encoding="utf-8")
print(f"Saved misses to {misses_path}")

print("\nSaving HF model & tokenizer...")
hf_dir = HF_SAVE_DIR
tokenizer.save_pretrained(str(hf_dir))
model.save_pretrained(str(hf_dir))

print("Saving classifier with joblib...")
joblib.dump(
    {
        "classifier": clf,
        "label_map": label_map,
        "threshold": float(best_threshold),
        "validation_threshold_metrics": best_val_info,
        "test_metrics_at_0_5": metrics_05,
        "test_metrics_at_chosen_threshold": metrics_best,
        "max_length": MAX_LENGTH,
        "model_name": MODEL_NAME,
    },
    str(CLF_PATH),
)

# Save threshold + metrics separately for easy inspection
threshold_info_path = CLF_PATH.with_name(CLF_PATH.stem + "_threshold_info.json")
with open(threshold_info_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "chosen_threshold": float(best_threshold),
            "validation_threshold_metrics": best_val_info,
            "test_metrics_at_0_5": metrics_05,
            "test_metrics_at_chosen_threshold": metrics_best,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"Saved classifier to {CLF_PATH}")
print(f"Saved threshold info to {threshold_info_path}")
print("Done.")