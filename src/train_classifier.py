import json
import os
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

from config import CLF_PATH, CSV_PATH, HF_SAVE_DIR, LOGS_DIR, MAX_LENGTH, MISSES_PATH, MODEL_NAME

RANDOM_STATE = 42
OPTIMIZATION_TARGET = os.getenv("TRAIN_TARGET", "f1").strip().lower()
MIN_SPAM_PRECISION = float(os.getenv("MIN_SPAM_PRECISION", "0.90"))
MIN_SPAM_RECALL = float(os.getenv("MIN_SPAM_RECALL", "0.93"))

if OPTIMIZATION_TARGET not in {"f1", "precision"}:
    raise ValueError("TRAIN_TARGET must be either 'f1' or 'precision'")

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"sklearn\.linear_model\._logistic",
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SOURCE = HF_SAVE_DIR if (HF_SAVE_DIR / "config.json").exists() else MODEL_NAME
LOCAL_ONLY = isinstance(MODEL_SOURCE, Path)

print(f"Loading HF model from: {MODEL_SOURCE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE, local_files_only=LOCAL_ONLY)
model = AutoModel.from_pretrained(MODEL_SOURCE, local_files_only=LOCAL_ONLY).to(device)
model.eval()


@torch.no_grad()
def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        hidden = int(getattr(model.config, "hidden_size", 312))
        return np.empty((0, hidden), dtype=np.float32)

    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts[start:start + batch_size]

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

        if batch_idx % 20 == 0 or batch_idx == total_batches:
            print(f"  Encoded batches: {batch_idx}/{total_batches}")

    return np.vstack(all_embeddings)


def load_dataset(path: Path) -> tuple[pd.DataFrame, dict[Any, int]]:
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")

    if "Message" not in df.columns or "Label" not in df.columns:
        raise ValueError("messages.csv должен содержать колонки 'Message' и 'Label'")

    label_map: dict[Any, int] = {
        "ham": 0,
        "spam": 1,
        0: 0,
        1: 1,
        "0": 0,
        "1": 1,
    }

    original_len = len(df)
    df["Message"] = (
        df["Message"]
        .astype(str)
        .str.replace("\r\n", "\n", regex=False)
        .str.replace("\r", "\n", regex=False)
        .str.strip()
    )
    df = df[df["Message"] != ""]
    non_empty_len = len(df)

    df["y"] = df["Label"].map(label_map)
    df = df.dropna(subset=["Message", "y"]).copy()
    mapped_len = len(df)
    df["y"] = df["y"].astype(int)

    dedup_before = len(df)
    df = df.drop_duplicates(subset=["Message", "y"]).reset_index(drop=True)
    dedup_after = len(df)

    print(f"Rows read: {original_len}")
    print(f"Rows after removing empty text: {non_empty_len}")
    print(f"Rows after label mapping: {mapped_len}")
    print(f"Removed exact duplicates: {dedup_before - dedup_after}")
    print(f"Final rows: {len(df)}")

    spam_count = int(df["y"].sum())
    ham_count = int(len(df) - spam_count)
    print(f"Class balance -> spam: {spam_count}, ham: {ham_count}")

    return df, label_map


def build_model(config: dict[str, Any]) -> LogisticRegression:
    kwargs: dict[str, Any] = {
        "max_iter": 5000,
        "random_state": RANDOM_STATE,
        "C": config["C"],
        "class_weight": config["class_weight"],
        "solver": config["solver"],
        "penalty": config["penalty"],
    }
    return LogisticRegression(**kwargs)


def evaluate_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_proba >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "ham_precision": float(prec[0]),
        "ham_recall": float(rec[0]),
        "ham_f1": float(f1[0]),
        "spam_precision": float(prec[1]),
        "spam_recall": float(rec[1]),
        "spam_f1": float(f1[1]),
        "ham_support": int(support[0]),
        "spam_support": int(support[1]),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def pick_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    candidates = np.arange(0.30, 0.91, 0.01)

    best_metrics: dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None

    for thr in candidates:
        metrics = evaluate_threshold(y_true, y_proba, float(thr))
        if OPTIMIZATION_TARGET == "precision":
            meets_recall = int(metrics["spam_recall"] >= MIN_SPAM_RECALL)
            key = (
                meets_recall,
                metrics["spam_precision"],
                metrics["spam_f1"],
                metrics["spam_recall"],
                metrics["ham_f1"],
                -metrics["fp"],
            )
        else:
            meets_precision = int(metrics["spam_precision"] >= MIN_SPAM_PRECISION)
            key = (
                meets_precision,
                metrics["spam_f1"],
                metrics["spam_recall"],
                metrics["ham_f1"],
                -metrics["fp"],
            )

        if best_key is None or key > best_key:
            best_key = key
            best_metrics = metrics

    assert best_metrics is not None
    return best_metrics


def main() -> None:
    df, label_map = load_dataset(CSV_PATH)

    texts = df["Message"].astype(str).tolist()
    y = df["y"].to_numpy(dtype=int)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        texts,
        y,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1764705882,  # 0.15 overall
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    print(
        f"Split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )

    print("Encoding train texts...")
    X_train_emb = encode_texts(X_train)

    print("Encoding val texts...")
    X_val_emb = encode_texts(X_val)

    print("Encoding test texts...")
    X_test_emb = encode_texts(X_test)

    search_space: list[dict[str, Any]] = []
    Cs = [0.08, 0.12, 0.2, 0.35, 0.6, 1.0, 1.8, 3.0, 5.0]
    class_weights: list[Any] = [
        None,
        "balanced",
        {0: 1.0, 1: 0.6},
        {0: 1.0, 1: 0.8},
        {0: 1.0, 1: 1.2},
        {0: 1.0, 1: 1.5},
        {0: 1.0, 1: 1.8},
        {0: 1.0, 1: 2.2},
    ]

    for c in Cs:
        for cw in class_weights:
            search_space.append({"solver": "lbfgs", "penalty": "l2", "C": c, "class_weight": cw})
            search_space.append({"solver": "liblinear", "penalty": "l2", "C": c, "class_weight": cw})
            search_space.append({"solver": "liblinear", "penalty": "l1", "C": c, "class_weight": cw})

    print(f"Hyperparameter candidates: {len(search_space)}")
    print(f"Optimization target: {OPTIMIZATION_TARGET}")
    if OPTIMIZATION_TARGET == "precision":
        print(f"Recall floor for selection: {MIN_SPAM_RECALL:.2f}")
    else:
        print(f"Precision floor for selection: {MIN_SPAM_PRECISION:.2f}")

    best_result: dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None
    search_results: list[dict[str, Any]] = []

    for idx, config in enumerate(search_space, start=1):
        clf = build_model(config)
        clf.fit(X_train_emb, y_train)

        val_proba = clf.predict_proba(X_val_emb)[:, 1]
        val_metrics = pick_best_threshold(y_val, val_proba)

        if OPTIMIZATION_TARGET == "precision":
            meets_recall = int(val_metrics["spam_recall"] >= MIN_SPAM_RECALL)
            key = (
                meets_recall,
                val_metrics["spam_precision"],
                val_metrics["spam_f1"],
                val_metrics["spam_recall"],
                val_metrics["ham_f1"],
                -val_metrics["fp"],
            )
        else:
            meets_precision = int(val_metrics["spam_precision"] >= MIN_SPAM_PRECISION)
            key = (
                meets_precision,
                val_metrics["spam_f1"],
                val_metrics["spam_recall"],
                val_metrics["ham_f1"],
                -val_metrics["fp"],
            )

        result = {
            "config": config,
            "val_metrics": val_metrics,
            "selection_key": key,
        }
        search_results.append(result)

        if best_key is None or key > best_key:
            best_key = key
            best_result = result

        if idx % 15 == 0 or idx == len(search_space):
            print(f"  Trained candidates: {idx}/{len(search_space)}")

    assert best_result is not None

    best_config = best_result["config"]
    best_threshold = float(best_result["val_metrics"]["threshold"])

    print("\n=== Best validation config ===")
    print(best_config)
    print(f"Threshold: {best_threshold:.2f}")
    print(
        "Val spam metrics: "
        f"precision={best_result['val_metrics']['spam_precision']:.4f}, "
        f"recall={best_result['val_metrics']['spam_recall']:.4f}, "
        f"f1={best_result['val_metrics']['spam_f1']:.4f}"
    )

    print("\nRetraining best model on train+val...")
    X_train_val_emb = np.vstack([X_train_emb, X_val_emb])
    y_train_val_np = np.concatenate([y_train, y_val])

    final_clf = build_model(best_config)
    final_clf.fit(X_train_val_emb, y_train_val_np)

    test_proba = final_clf.predict_proba(X_test_emb)[:, 1]
    test_metrics = evaluate_threshold(y_test, test_proba, best_threshold)
    y_test_pred = (test_proba >= best_threshold).astype(int)

    print("\n=== Final evaluation on test split ===")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(
        "SPAM -> "
        f"precision={test_metrics['spam_precision']:.4f}, "
        f"recall={test_metrics['spam_recall']:.4f}, "
        f"f1={test_metrics['spam_f1']:.4f}"
    )
    print(
        "HAM  -> "
        f"precision={test_metrics['ham_precision']:.4f}, "
        f"recall={test_metrics['ham_recall']:.4f}, "
        f"f1={test_metrics['ham_f1']:.4f}"
    )

    cm = np.array([[test_metrics["tn"], test_metrics["fp"]], [test_metrics["fn"], test_metrics["tp"]]])
    print("\nConfusion matrix [rows=true, cols=pred]:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, digits=4))

    miss_records = []
    for text, true_label, pred_label, proba in zip(X_test, y_test, y_test_pred, test_proba):
        if int(pred_label) != int(true_label):
            miss_type = "FN" if int(true_label) == 1 else "FP"
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

    misses_df = pd.DataFrame(miss_records)
    misses_df.to_csv(MISSES_PATH, index=False, encoding="utf-8")
    print(f"Saved misses: {len(miss_records)} -> {MISSES_PATH}")

    print("\nSaving HF model & tokenizer...")
    tokenizer.save_pretrained(HF_SAVE_DIR)
    model.save_pretrained(HF_SAVE_DIR)

    print("Saving classifier artifact...")
    artifact = {
        "classifier": final_clf,
        "label_map": label_map,
        "threshold": float(best_threshold),
        "training": {
            "random_state": RANDOM_STATE,
            "selection_policy": (
                f"maximize spam_precision with spam_recall>={MIN_SPAM_RECALL:.2f} on validation"
                if OPTIMIZATION_TARGET == "precision"
                else f"maximize spam_f1 with spam_precision>={MIN_SPAM_PRECISION:.2f} on validation"
            ),
            "optimization_target": OPTIMIZATION_TARGET,
            "best_config": best_config,
            "val_metrics": best_result["val_metrics"],
            "test_metrics": test_metrics,
            "dataset_rows": int(len(df)),
            "split_sizes": {
                "train": int(len(X_train)),
                "val": int(len(X_val)),
                "test": int(len(X_test)),
            },
        },
    }
    joblib.dump(artifact, CLF_PATH)
    print(f"Saved classifier to {CLF_PATH}")

    report = {
        "optimization_target": OPTIMIZATION_TARGET,
        "best_config": best_config,
        "best_threshold": float(best_threshold),
        "val_metrics": best_result["val_metrics"],
        "test_metrics": test_metrics,
        "top_candidates": [
            {
                "config": item["config"],
                "val_metrics": item["val_metrics"],
            }
            for item in sorted(
                search_results,
                key=lambda x: x["selection_key"],
                reverse=True,
            )[:10]
        ],
    }

    report_path = LOGS_DIR / "training_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved report to {report_path}")

    print("Done.")


if __name__ == "__main__":
    main()
