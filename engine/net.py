"""Small HTTP/JSON helper shared by the live air-quality and forecast clients.

Every transport, HTTP and decode failure is normalised into ``LiveDataError`` so
callers have exactly one exception type to reason about. Transport-level failures
(resets, timeouts, DNS) are retried with exponential backoff -- data.gov.in
in particular drops connections under load -- but a retry never invents a value:
the caller gets either a real response or an error.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Optional

HTTP_TIMEOUT_S = 45.0
HTTP_RETRIES = 3
HTTP_RETRY_BACKOFF_S = 1.5
USER_AGENT = "Vayumitre/1.0 (+live telemetry client)"


class LiveDataError(RuntimeError):
    """A live feed could not supply real, current data."""


def get_json(url: str, *, headers: Optional[dict] = None, timeout: float = HTTP_TIMEOUT_S,
             label: str = "feed", retries: int = HTTP_RETRIES) -> dict:
    request = urllib.request.Request(url, method="GET")
    request.add_header("Accept", "application/json")
    # data.gov.in silently stalls requests that carry no User-Agent.
    request.add_header("User-Agent", USER_AGENT)
    for key, value in (headers or {}).items():
        request.add_header(key, value)

    last_error = ""
    attempts = max(1, retries)
    for attempt in range(attempts):
        if attempt:
            time.sleep(HTTP_RETRY_BACKOFF_S * (2 ** (attempt - 1)))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise LiveDataError(f"{label}: HTTP {exc.code} {exc.reason}. {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(getattr(exc, "reason", None) or exc)
            continue
        try:
            return json.loads(body)
        except ValueError as exc:
            raise LiveDataError(f"{label}: response was not valid JSON ({body[:160]!r}).") from exc

    raise LiveDataError(f"{label}: unreachable after {attempts} attempt(s) ({last_error}).")
