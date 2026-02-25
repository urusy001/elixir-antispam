import re
from pathlib import Path
from typing import IO

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

_OCR_CONFIGS = (
    "--oem 3 --psm 6 -c preserve_interword_spaces=1",
    "--oem 3 --psm 11",
)
_NON_TEXT_LINE_RE = re.compile(r"^[^\wА-Яа-яЁё]+$", re.UNICODE)
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]{2,}")
_FINGERPRINT_RE = re.compile(r"[^0-9a-zа-яё]+", re.IGNORECASE)


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def _resize_for_ocr(image: Image.Image) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest >= 1800:
        return image

    if longest < 800:
        scale = 3
    elif longest < 1400:
        scale = 2
    else:
        scale = 1.5

    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _otsu_threshold(gray_image: Image.Image) -> int:
    hist = gray_image.histogram()[:256]
    total = sum(hist)
    if total == 0:
        return 127

    sum_total = sum(index * count for index, count in enumerate(hist))
    sum_background = 0
    weight_background = 0
    best_threshold = 127
    max_variance = -1.0

    for threshold, count in enumerate(hist):
        weight_background += count
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += threshold * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        between_class_variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            best_threshold = threshold

    return best_threshold


def _binarize(gray_image: Image.Image) -> Image.Image:
    threshold = _otsu_threshold(gray_image)
    return gray_image.point(lambda value: 255 if value >= threshold else 0).convert("L")


def _build_variants(image: Image.Image) -> list[Image.Image]:
    base = _resize_for_ocr(_flatten_to_rgb(image))
    photo = ImageEnhance.Sharpness(base).enhance(1.6)

    gray = ImageOps.grayscale(photo)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.4)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    gray = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=2))

    binary = _binarize(gray)
    inverted_binary = ImageOps.invert(binary)

    return [photo, gray, binary, inverted_binary]


def _normalize_ocr_text(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        if _NON_TEXT_LINE_RE.fullmatch(line):
            continue

        alnum_count = sum(char.isalnum() for char in line)
        if alnum_count < 2:
            continue
        if len(line) >= 8 and alnum_count / len(line) < 0.3:
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _score_ocr_text(text: str) -> int:
    if not text:
        return -10_000

    alnum_count = sum(char.isalnum() for char in text)
    cyrillic_count = len(_CYRILLIC_RE.findall(text))
    word_count = len(_WORD_RE.findall(text))
    penalty = sum(text.count(symbol) for symbol in "|`~")
    return alnum_count + cyrillic_count * 2 + word_count * 4 - penalty * 3


def _line_fingerprint(line: str) -> str:
    return _FINGERPRINT_RE.sub("", line.lower())


def _merge_short_candidates(texts: list[str]) -> str:
    merged_lines: list[str] = []
    seen_fingerprints: set[str] = set()

    for text in texts:
        for line in text.splitlines():
            fingerprint = _line_fingerprint(line)
            if not fingerprint:
                continue
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            merged_lines.append(line)

    return "\n".join(merged_lines).strip()


def extract_text_from_image(image_path: Path | IO[bytes]) -> str:
    """Run Tesseract OCR on a photo and return improved Russian/English text."""
    image = Image.open(image_path)
    variants = _build_variants(image)

    candidates: list[tuple[int, str]] = []
    seen_texts: set[str] = set()

    for variant in variants:
        for config in _OCR_CONFIGS:
            raw_text = pytesseract.image_to_string(variant, lang="rus+eng", config=config)
            text = _normalize_ocr_text(raw_text)
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            candidates.append((_score_ocr_text(text), text))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_text = candidates[0][1]
    if len(best_text) >= 20 or len(candidates) == 1:
        return best_text

    return _merge_short_candidates([text for _, text in candidates[:3]])
