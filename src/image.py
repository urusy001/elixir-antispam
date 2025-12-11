from __future__ import annotations

import pytesseract
from typing import IO
from pathlib import Path
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

def extract_text_from_image(image_path: Path | IO[bytes]) -> str:
    """Run Tesseract OCR on a single image and return text (Russian + English)."""
    image = Image.open(image_path)
    print('extract_text_from_image')
    return pytesseract.image_to_string(image, lang="rus+eng")
