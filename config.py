import os
from os import getenv
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

WORKING_DIR=Path(__file__).parent
LOGS_DIR=WORKING_DIR / "logs"
SRC_DIR=WORKING_DIR / "src"
CSV_PATH=LOGS_DIR / "messages.csv"
MISSES_PATH=LOGS_DIR / "misses.csv"
HF_SAVE_DIR=SRC_DIR / "hf_rubert_spam"
CLF_PATH=SRC_DIR / "spam_classifier.joblib"
MAX_LENGTH=128
MODEL_NAME="cointegrated/rubert-tiny"

CAPTCHA_MAX_ATTEMPTS = 3
POLL_TIMEOUT_SECONDS = 30
POLL_TIMEOUT_BUFFER = 3

MOSCOW_TZ=ZoneInfo("Europe/Moscow")
ANTISPAM_BOT_TOKEN=getenv("ANTISPAM_BOT_TOKEN")
ELIXIR_CHAT_ID=int(getenv("ELIXIR_CHAT_ID"))
REPORTS_CHANNEL_ID=int(getenv("REPORTS_CHANNEL_ID"))

POSTGRES_USER=getenv("POSTGRES_USER")
POSTGRES_PASSWORD=getenv("POSTGRES_PASSWORD")
POSTGRES_DB=getenv("POSTGRES_DB")
POSTGRES_HOST=getenv("POSTGRES_HOST")
POSTGRES_PORT=getenv("POSTGRES_PORT")

SYNC_DATABASE_URL=f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
ASYNC_DATABASE_URL=f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
