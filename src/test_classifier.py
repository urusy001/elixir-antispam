import asyncio
import os
import re
from dataclasses import dataclass
from functools import partial

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
BASE_THRESHOLD = float(artifact.get("threshold", 0.65))
if BASE_THRESHOLD <= 0.0 or BASE_THRESHOLD >= 1.0:
    BASE_THRESHOLD = 0.65

URL_RE = re.compile(r"(https?://|www\.|t\.me/|telegram\.me/)", re.IGNORECASE)
TG_HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{4,}")
MONEY_RE = re.compile(
    r"(?:\$|€|₽)\s?\d{2,6}|\d{2,6}\s?(?:\$|€|₽|руб|rur|usd|usdt|btc)",
    re.IGNORECASE,
)

SPAM_KEYWORDS = (
    "удален",
    "удаленно",
    "удаленка",
    "доход",
    "прибыл",
    "заработ",
    "без опыта",
    "набор в команду",
    "инвест",
    "крипт",
    "bybit",
    "binance",
    "p2p",
    "ставки",
    "казино",
    "пассивный доход",
    "выплаты",
    "закрытый чат",
)

CTA_KEYWORDS = (
    "пиши в лс",
    "пишите в лс",
    "в личку",
    "в личные сообщения",
    "пиши +",
    "пишите +",
    "жду в лс",
    "за подробностями",
)

HAM_HINTS = (
    "подскаж",
    "вопрос",
    "как ",
    "когда ",
    "почему ",
    "где ",
    "спасибо",
    "заказ",
    "доставка",
    "оплат",
    "состав",
    "дозиров",
    "картридж",
)


@dataclass(frozen=True)
class SpamAnalysis:
    is_spam: bool
    final_probability: float
    model_probability: float
    heuristic_probability: float
    threshold_used: float
    reasons: tuple[str, ...]


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("\n", " ").split())


def _caps_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    caps = sum(1 for ch in letters if ch.isupper())
    return caps / len(letters)


def _effective_threshold(user_risk: float) -> float:
    bounded_risk = min(1.35, max(0.75, user_risk))
    # Risk > 1.0 => lower threshold (stricter); risk < 1.0 => raise threshold.
    threshold = BASE_THRESHOLD - (bounded_risk - 1.0) * 0.10
    return min(0.90, max(0.45, threshold))


def _heuristic_probability(text: str) -> tuple[float, tuple[str, ...]]:
    normalized = _normalize(text)
    reasons: list[str] = []
    score = 0.0

    handle_matches = TG_HANDLE_RE.findall(text)
    if handle_matches:
        score += min(0.25, 0.12 + 0.06 * (len(handle_matches) - 1))
        reasons.append("telegram_handle")

    has_url = bool(URL_RE.search(text))
    if has_url:
        score += 0.16
        reasons.append("url")

    if MONEY_RE.search(text):
        score += 0.14
        reasons.append("money_claim")

    keyword_hits = sum(1 for kw in SPAM_KEYWORDS if kw in normalized)
    if keyword_hits:
        score += min(0.32, keyword_hits * 0.06)
        reasons.append("spam_keywords")

    cta_hits = sum(1 for kw in CTA_KEYWORDS if kw in normalized)
    if cta_hits:
        score += min(0.18, cta_hits * 0.09)
        reasons.append("call_to_action")

    if normalized.count("срочно") >= 2:
        score += 0.08
        reasons.append("urgent_repetition")

    if ("!!!" in text or "???" in text) and len(text) >= 25:
        score += 0.07
        reasons.append("repeated_punctuation")

    if _caps_ratio(text) >= 0.55 and len(text) >= 25:
        score += 0.10
        reasons.append("caps_shouting")

    # Ham dampening to reduce obvious false positives in support-style messages.
    ham_hits = sum(1 for kw in HAM_HINTS if kw in normalized)
    if ham_hits and not has_url and not handle_matches:
        score -= min(0.14, ham_hits * 0.04)
        reasons.append("ham_context")

    if "?" in text and not has_url and not handle_matches and len(text) <= 200:
        score -= 0.05
        reasons.append("question_like")

    score = min(1.0, max(0.0, score))
    return score, tuple(reasons)


def _load_eval_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if "Message" not in df.columns or "Label" not in df.columns:
        raise ValueError("messages.csv должен содержать колонки 'Message' и 'Label'")
    return df


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
    cls_emb = outputs.last_hidden_state[:, 0, :]  # (1, hidden)
    return cls_emb.cpu().numpy()


def analyze_spam_sync(text: str, *, user_risk: float = 1.0) -> SpamAnalysis:
    text = (text or "").strip()
    threshold = _effective_threshold(user_risk)
    if not text:
        return SpamAnalysis(
            is_spam=False,
            final_probability=0.0,
            model_probability=0.0,
            heuristic_probability=0.0,
            threshold_used=threshold,
            reasons=("empty_text",),
        )

    emb = embed_one(text)
    model_probability = float(clf.predict_proba(emb)[0, 1])
    heuristic_probability, reasons = _heuristic_probability(text)

    final_probability = model_probability * 0.82 + heuristic_probability * 0.18
    if model_probability < 0.50 and heuristic_probability >= 0.60:
        final_probability = max(
            final_probability,
            min(1.0, model_probability + heuristic_probability * 0.25),
        )

    final_probability = min(1.0, max(0.0, final_probability))
    is_spam_flag = final_probability >= threshold

    return SpamAnalysis(
        is_spam=is_spam_flag,
        final_probability=final_probability,
        model_probability=model_probability,
        heuristic_probability=heuristic_probability,
        threshold_used=threshold,
        reasons=reasons,
    )


def is_spam_sync(text: str, *, user_risk: float = 1.0) -> tuple[bool, float]:
    analysis = analyze_spam_sync(text, user_risk=user_risk)
    return analysis.is_spam, analysis.final_probability


async def is_spam(text: str, *, user_risk: float = 1.0) -> tuple[bool, float]:
    loop = asyncio.get_running_loop()
    task = partial(is_spam_sync, text, user_risk=user_risk)
    return await loop.run_in_executor(None, task)

def main() -> None:
    # 1. Читаем test.csv
    path = LOGS_DIR / "messages.csv"
    print(f"[INFO] Loading test data from: {path}")
    df = _load_eval_dataset(path)

    if "Message" not in df.columns or "Label" not in df.columns:
        raise ValueError("test.csv должен содержать колонки 'Message' и 'Label'")

    # поддерживаем 0/1 и 'ham'/'spam'
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

    # 2. Прогоняем через модель
    preds = []
    probs = []
    model_probs = []
    heuristic_probs = []
    thresholds = []
    reasons = []

    for i, text in enumerate(texts, 1):
        analysis = analyze_spam_sync(text)
        preds.append(int(analysis.is_spam))
        probs.append(analysis.final_probability)
        model_probs.append(analysis.model_probability)
        heuristic_probs.append(analysis.heuristic_probability)
        thresholds.append(analysis.threshold_used)
        reasons.append("|".join(analysis.reasons))
        if i % 100 == 0 or i == len(texts):
            print(f"[INFO] Processed {i}/{len(texts)}")

    preds = np.array(preds, dtype=int)
    probs = np.array(probs, dtype=float)

    # 3. Метрики
    print("\n=== Confusion matrix (true rows / predicted cols) ===")
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    print(cm)

    print("\n=== Classification report ===")
    print(classification_report(y_true, preds, digits=4))

    acc = accuracy_score(y_true, preds)
    print(f"\nAccuracy: {acc:.4f}")

    # 4. Сохраняем файл с предсказаниями
    df["pred"] = preds
    df["proba_spam"] = probs
    df["proba_model"] = model_probs
    df["proba_heuristic"] = heuristic_probs
    df["threshold_used"] = thresholds
    df["reasons"] = reasons
    out_path = LOGS_DIR / "test_with_preds.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
