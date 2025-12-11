import os
import torch
import numpy as np
import joblib
import asyncio

from transformers import AutoTokenizer, AutoModel

from config import HF_SAVE_DIR, CLF_PATH, MAX_LENGTH

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(HF_SAVE_DIR)
model = AutoModel.from_pretrained(HF_SAVE_DIR).to(device)
model.eval()
artifact = joblib.load(CLF_PATH)
clf = artifact["classifier"]
threshold = .52

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


def is_spam_sync(text: str) -> tuple[bool, float]:
    emb = embed_one(text)
    proba_spam = float(clf.predict_proba(emb)[0, 1])
    return proba_spam >= threshold, proba_spam

async def is_spam(text: str) -> tuple[bool, float]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, is_spam_sync, text)
