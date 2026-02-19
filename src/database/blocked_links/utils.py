import re

from urllib.parse import urlparse
from typing import Optional

_TWO_LEVEL_SUFFIXES = {
    "co.uk",
    "org.uk",
    "gov.uk",
    "ac.uk",
    "com.au",
    "net.au",
    "org.au",
    "co.nz",
    "com.br",
    "com.tr",
    "com.ua",
    "co.jp",
    "co.in",
    "co.kr",
    "com.sg",
    "com.mx",
    "com.ar",
}

_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>()]+")
_BARE_DOMAIN_RE = re.compile(r"(?i)\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}\b")
_BARE_TELEGRAM_LINK_RE = re.compile(r"(?i)\b(?:t\.me|telegram\.me)/[^\s<>()]+")
_IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_TELEGRAM_HOSTS = {"t.me", "telegram.me"}


def extract_base_domain(value: str) -> Optional[str]:
    candidate = value.strip()
    if not candidate:
        return None

    parsed = urlparse(candidate if "://" in candidate else f"//{candidate}")
    host = (parsed.hostname or candidate.split("/")[0]).strip().lower().strip(".")
    if not host or _IPV4_RE.fullmatch(host):
        return None

    if host.startswith("www."):
        host = host[4:]

    labels = [part for part in host.split(".") if part]
    if len(labels) < 2:
        return None

    suffix = ".".join(labels[-2:])
    if len(labels) >= 3 and suffix in _TWO_LEVEL_SUFFIXES:
        return ".".join(labels[-3:])

    return ".".join(labels[-2:])


def normalize_blocked_link(value: str) -> Optional[str]:
    candidate = value.strip()
    if not candidate:
        return None

    parsed = urlparse(candidate if "://" in candidate else f"//{candidate}")
    host = (parsed.hostname or "").strip().lower().strip(".")
    if host.startswith("www."):
        host = host[4:]

    if host in _TELEGRAM_HOSTS:
        path = (parsed.path or "").strip().strip("/")
        if path:
            return f"t.me/{path}".lower()

    return extract_base_domain(candidate)


def extract_blocked_targets_from_text(text: str) -> set[str]:
    if not text:
        return set()

    targets: set[str] = set()
    for pattern in (_URL_RE, _BARE_TELEGRAM_LINK_RE, _BARE_DOMAIN_RE):
        for match in pattern.finditer(text):
            candidate = match.group(0).strip(" \n\r\t<>[](){}\"'.,!?;:")
            blocked_target = normalize_blocked_link(candidate)
            if not blocked_target:
                continue

            targets.add(blocked_target)
            base_domain = extract_base_domain(candidate)
            if base_domain:
                targets.add(base_domain)

    return targets


def extract_base_domains_from_text(text: str) -> set[str]:
    if not text:
        return set()

    domains: set[str] = set()
    for pattern in (_URL_RE, _BARE_DOMAIN_RE):
        for match in pattern.finditer(text):
            candidate = match.group(0).strip(" \n\r\t<>[](){}\"'.,!?;:")
            domain = extract_base_domain(candidate)
            if domain:
                domains.add(domain)

    return domains
