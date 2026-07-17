from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from typing import Mapping

import requests

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(frozen=True)
class HttpClient:
    timeout: int = 30
    retries: int = 2
    pause_seconds: float = 1.0

    def get_text(self, url: str, headers: Mapping[str, str] | None = None) -> str:
        response = self._get(url, headers=headers)
        return response.text

    def get_json(self, url: str, headers: Mapping[str, str] | None = None) -> dict:
        response = self._get(url, headers=headers)
        return response.json()

    def _get(self, url: str, headers: Mapping[str, str] | None = None) -> requests.Response:
        merged_headers = {**DEFAULT_HEADERS, **(dict(headers or {}))}
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.get(url, headers=merged_headers, timeout=self.timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.retries:
                    sleep(self.pause_seconds * (attempt + 1))
        raise RuntimeError(f"GET failed after {self.retries + 1} attempts: {url}") from last_error
