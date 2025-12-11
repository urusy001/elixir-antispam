import asyncio
import logging
import sys

from src.bot.main import run_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)

if __name__ == '__main__':
    try: asyncio.run(run_bot())
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
        sys.exit(0)
