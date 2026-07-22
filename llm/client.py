"""OpenAI-compatible chat client.

Any endpoint that speaks the OpenAI ``/v1/chat/completions`` protocol
works — official OpenAI, Azure OpenAI, local OpenAI-compatible servers,
vLLM / Ollama in OpenAI-compat mode, etc. The provider difference is
just ``api_base`` + ``api_key`` + ``model``, all supplied by
:class:`llm.config.LLMConfig`.

The client is intentionally thin: no retries, no streaming, no tool
calling. Add those upstream if needed — the goal here is a single,
predictable choke point that the rest of the app can mock in tests.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm.config import LLMConfig, mask_key

# openai>=1.0 exposes an OpenAI class that accepts base_url — that's
# what makes it "provider-agnostic". Import guarded so the app still
# runs if the SDK isn't installed.
try:
    from openai import OpenAI  # type: ignore
    _OPENAI_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _exc:  # noqa: BLE001
    OpenAI = None  # type: ignore
    _OPENAI_IMPORT_ERROR = _exc

# httpx is a transitive dep of openai — importable whenever OpenAI is.
# We only need it to build a custom-verified transport for private
# endpoints with an internal CA.
try:
    import httpx  # type: ignore
except Exception:  # noqa: BLE001
    httpx = None  # type: ignore


# The safe fallback we hand back when JSON parsing gives up entirely.
# Kept as a module constant so callers can `is`-compare and detect it.
FALLBACK_FORECAST: Dict[str, Any] = {
    "probability_yes": 0.5,
    "confidence": 0.3,
    "reasoning_summary": "LLM returned invalid JSON; fallback used.",
    "key_evidence": [],
    "risk_factors": ["Invalid JSON response"],
}


# Match the first {...} block. Non-greedy on the outside so we don't
# swallow trailing prose, but we still need balance-aware extraction
# below because JSON can nest.
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_substring(text: str) -> Optional[str]:
    """Best-effort JSON extraction from a mixed text/JSON response.

    Some providers wrap the JSON in prose or code fences. We scan for a
    balanced ``{...}`` block and return it. Returns None if no plausible
    block is found.
    """
    if not text:
        return None

    # Strip ```json ... ``` fences if present — common LLM output.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)

    # Balance-aware scan for the first complete top-level object.
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start : i + 1]

    # Last resort: regex, may be lossy if there are stray '{' in prose.
    m = _JSON_BLOCK_RE.search(text)
    return m.group(0) if m else None


class LLMClient:
    """Thin wrapper around ``openai.OpenAI`` bound to an :class:`LLMConfig`.

    Attributes:
        config:     the frozen config used to build the client.
        available:  True iff the SDK loaded AND a key is configured.
                    Callers should branch on this before ``chat``/``chat_json``
                    — otherwise those methods raise ``RuntimeError``.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client: Optional[OpenAI] = None  # type: ignore[assignment]
        self._unavailable_reason: Optional[str] = None

        if OpenAI is None:
            self._unavailable_reason = (
                f"openai package not importable: {_OPENAI_IMPORT_ERROR!r}"
            )
            return
        if not self.config.is_configured:
            self._unavailable_reason = "no API key configured"
            return

        try:
            # base_url is what makes this provider-agnostic — omit it to
            # get the OpenAI default, pass it to hit Azure/vLLM/Ollama.
            kwargs: Dict[str, Any] = {
                "api_key": self.config.api_key,
                "timeout": self.config.timeout,
            }
            if self.config.api_base:
                kwargs["base_url"] = self.config.api_base

            # Private endpoints may use an internal CA that certifi
            # doesn't ship. httpx ignores
            # SSL_CERT_FILE, so we build our own client with the right
            # verify= value and hand it to the SDK.
            http_client = self._build_http_client()
            if http_client is not None:
                kwargs["http_client"] = http_client

            self._client = OpenAI(**kwargs)
        except Exception as exc:  # noqa: BLE001
            self._unavailable_reason = f"failed to init OpenAI client: {exc}"
            self._client = None

    def _build_http_client(self):
        """Return an httpx.Client honoring the CA bundle / verify flag.

        Returns None if httpx isn't importable or nothing needs to
        change from the SDK default. openai>=1.x accepts an
        ``http_client=`` kwarg on ``OpenAI()`` for exactly this purpose.

        Robustness note: if a user-provided ``ca_bundle`` fails to
        verify the endpoint (common on private networks where the
        bundle contains an intermediate CA but not the root), we
        transparently retry once against the auto-detected system CA
        store — that's what ``curl`` effectively does when ``--cacert``
        is combined with the OS trust anchor set. This turns an opaque
        ``unable to get issuer certificate`` into a working request.
        """
        if httpx is None:
            return None
        if self.config.ssl_verify and not self.config.ca_bundle:
            return None

        verify: Any = self.config.ca_bundle if self.config.ca_bundle else self.config.ssl_verify

        # Health-check the CA bundle against the configured api_base
        # BEFORE handing the client to the SDK. Doing it here means
        # errors surface at init time, not deep inside the SDK's first
        # request, and gives us a chance to fall back gracefully.
        if self.config.ca_bundle and self.config.api_base:
            if not self._probe_verify(verify):
                fallback = self._system_ca_fallback()
                if fallback and self._probe_verify(fallback):
                    print(
                        f"[LLMClient] custom CA {self.config.ca_bundle!r} "
                        f"could not verify {self.config.api_base}; falling "
                        f"back to system CA {fallback!r}",
                        file=sys.stderr,
                    )
                    verify = fallback
                    # keep ``config.ca_bundle`` untouched so describe()
                    # still shows what the user asked for — the swap is
                    # visible in stderr, not silently rewritten.

        try:
            return httpx.Client(verify=verify, timeout=self.config.timeout)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[LLMClient] failed to build custom httpx client "
                f"(verify={verify!r}): {exc}; falling back to SDK default",
                file=sys.stderr,
            )
            return None

    def _probe_verify(self, verify: Any) -> bool:
        """One-shot TLS probe against ``api_base``.

        Any non-SSL failure (HTTP 4xx/5xx, connection refused, timeout)
        counts as "SSL passed" — we only care whether the trust chain
        validates. Returns True on success or non-SSL failure, False
        only when SSL verification itself fails.
        """
        try:
            with httpx.Client(verify=verify, timeout=5.0) as probe:
                probe.get(self.config.api_base, timeout=5.0)
            return True
        except httpx.ConnectError as exc:
            # ssl.SSLCertVerificationError arrives wrapped inside a
            # ConnectError; check the message rather than the type.
            msg = str(exc).lower()
            if "certificate" in msg or "ssl" in msg or "verify" in msg:
                return False
            return True                     # non-TLS connect failure — trust chain wasn't the blocker
        except Exception:                    # noqa: BLE001
            return True                     # HTTP-level errors are fine here

    def _system_ca_fallback(self) -> Optional[str]:
        """Return the first system CA bundle that exists, or None."""
        for p in (
            "/etc/ssl/certs/ca-certificates.crt",
            "/etc/pki/tls/certs/ca-bundle.crt",
            "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
            "/etc/ssl/cert.pem",
        ):
            try:
                if Path(p).is_file():
                    return p
            except Exception:  # noqa: BLE001
                pass
        return None

    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def unavailable_reason(self) -> Optional[str]:
        return self._unavailable_reason

    def describe(self) -> Dict[str, Any]:
        """Log-safe description — no raw key ever included."""
        d = self.config.public_dict()
        d["available"] = self.available
        d["unavailable_reason"] = self._unavailable_reason
        return d

    # ------------------------------------------------------------------
    # Core chat surface

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a chat request; return the assistant's text content.

        Raises RuntimeError if the client isn't ``available`` — callers
        should check first and fall back if needed.
        """
        if self._client is None:
            raise RuntimeError(
                f"LLMClient not available: {self._unavailable_reason}"
            )

        temp = self.config.temperature if temperature is None else temperature
        tok = self.config.max_tokens if max_tokens is None else max_tokens

        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temp,
        }
        if tok is not None:
            kwargs["max_tokens"] = tok

        completion = self._client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or ""

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send a chat request expecting strict JSON; parse defensively.

        Order of attempts:
          1. Request with ``response_format={"type":"json_object"}``.
             If the provider rejects that param (some OpenAI-compat
             servers don't support it), retry without it.
          2. ``json.loads`` on the raw content.
          3. Extract the first balanced ``{...}`` block and parse that.
          4. Give up: return a copy of :data:`FALLBACK_FORECAST`.
        """
        if self._client is None:
            # Consistent with chat(): don't silently return fallback for
            # a totally unconfigured client — surface it.
            raise RuntimeError(
                f"LLMClient not available: {self._unavailable_reason}"
            )

        temp = self.config.temperature if temperature is None else temperature
        tok = self.config.max_tokens if max_tokens is None else max_tokens

        base_kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temp,
        }
        if tok is not None:
            base_kwargs["max_tokens"] = tok

        content = ""
        try:
            completion = self._client.chat.completions.create(
                response_format={"type": "json_object"},
                **base_kwargs,
            )
            content = completion.choices[0].message.content or ""
        except TypeError:
            # SDK signature mismatch — very old client. Retry without.
            completion = self._client.chat.completions.create(**base_kwargs)
            content = completion.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            # Some OpenAI-compat servers 400 on response_format. Retry
            # once without it before giving up.
            msg = str(exc).lower()
            if "response_format" in msg or "unsupported" in msg or "400" in msg:
                try:
                    completion = self._client.chat.completions.create(**base_kwargs)
                    content = completion.choices[0].message.content or ""
                except Exception as exc2:  # noqa: BLE001
                    print(
                        f"[LLMClient] chat_json retry failed: {exc2}",
                        file=sys.stderr,
                    )
                    return dict(FALLBACK_FORECAST)
            else:
                print(f"[LLMClient] chat_json failed: {exc}", file=sys.stderr)
                return dict(FALLBACK_FORECAST)

        # --- parse ---
        try:
            return json.loads(content)
        except (ValueError, TypeError):
            pass

        extracted = _extract_json_substring(content)
        if extracted:
            try:
                return json.loads(extracted)
            except (ValueError, TypeError):
                pass

        print(
            f"[LLMClient] chat_json could not parse response "
            f"(model={self.config.model}, key={mask_key(self.config.api_key)}); "
            "returning fallback",
            file=sys.stderr,
        )
        return dict(FALLBACK_FORECAST)
