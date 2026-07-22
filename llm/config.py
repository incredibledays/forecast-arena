"""LLM configuration loader.

Load order (later sources override earlier ones only where explicitly
set — env is the primary channel because real API keys must never live
in a checked-in file):

    1. Built-in defaults (safe, all None / neutral).
    2. Environment variables (via python-dotenv if a .env file exists):
         LLM_PROVIDER, LLM_API_KEY, LLM_API_BASE, LLM_MODEL,
         LLM_TIMEOUT, LLM_TEMPERATURE, LLM_MAX_TOKENS
    3. Optional llm_providers.json in the project root:
         {
           "defaultProvider": "custom",
           "providers": {
             "custom": {
               "apiKeyEnv": "CUSTOM_LLM_API_KEY",
               "apiBase":   "https://llm.example.com/v1",
               "model":     "gpt-4o-mini"
             }
           }
         }
       The JSON file NEVER contains a raw API key — only the *name* of
       the env var that holds it (`apiKeyEnv`). This lets us commit the
       example file safely.

If nothing is configured the app remains runnable: `load_config()`
returns an `LLMConfig` with `api_key=None`, and `LLMClient.available`
is False. Callers must handle that case (fall back to a heuristic,
etc.).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001 — dotenv is optional at import time
    pass


# --- helpers ---------------------------------------------------------

def mask_key(api_key: Optional[str]) -> str:
    """Return a display-safe representation of an API key.

    Never returns more than the first 4 chars followed by ``****``. Empty
    / short keys are collapsed to a single placeholder so logs don't leak
    the fact that a short-lived stub key was used.
    """
    if not api_key:
        return "<none>"
    if len(api_key) <= 4:
        return "****"
    return f"{api_key[:4]}****"


def _env(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip()
    return v or None


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# --- config object ---------------------------------------------------

@dataclass
class LLMConfig:
    """All knobs the LLM layer needs, in one place.

    ``api_key`` is the only field that must be kept out of logs / UI.
    Use :func:`mask_key` when printing.
    """
    provider_name: str = "openai"
    api_key: Optional[str] = None
    api_base: Optional[str] = None          # e.g. https://api.openai.com/v1
    model: str = "gpt-4o-mini"
    timeout: float = 30.0
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    # Path to a CA bundle. Needed on private networks where the LLM
    # endpoint uses an internally-signed cert: certifi doesn't ship
    # private CAs, so we hand httpx an explicit
    # bundle. `None` means "use the SDK default" (certifi). Set the
    # LLM_CA_BUNDLE env var, or leave unset to auto-detect the system
    # store on Linux.
    ca_bundle: Optional[str] = None
    # Escape hatch — set LLM_SSL_VERIFY=false to disable TLS verification
    # entirely. INSECURE; only for local sandbox / debugging.
    ssl_verify: bool = True

    # Retained for debug / status endpoints. Not serialized by default.
    _source_notes: list = field(default_factory=list, repr=False)

    # --- convenience --------------------------------------------------

    @property
    def is_configured(self) -> bool:
        """True iff we have enough to actually hit an endpoint."""
        return bool(self.api_key) and bool(self.model)

    def public_dict(self) -> Dict[str, Any]:
        """Serializable, key-safe dict for logging / template display."""
        return {
            "provider_name": self.provider_name,
            "api_base": self.api_base,
            "model": self.model,
            "timeout": self.timeout,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key_masked": mask_key(self.api_key),
            "ca_bundle": self.ca_bundle,
            "ssl_verify": self.ssl_verify,
            "is_configured": self.is_configured,
        }


# --- loader ----------------------------------------------------------

# Candidate CA bundles on common Linux distributions. Ordered by how
# widely each is used; we take the first that exists. Users with a
# non-standard location can set LLM_CA_BUNDLE explicitly.
_SYSTEM_CA_CANDIDATES = (
    "/etc/ssl/certs/ca-certificates.crt",           # Debian/Ubuntu
    "/etc/pki/tls/certs/ca-bundle.crt",             # RHEL/CentOS/Fedora
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
    "/etc/ssl/cert.pem",                            # Alpine, macOS
)


def _detect_system_ca() -> Optional[str]:
    """Return the first system CA bundle we can find, or None."""
    for p in _SYSTEM_CA_CANDIDATES:
        if Path(p).is_file():
            return p
    return None


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() not in ("0", "false", "no", "off")


_DEFAULT_JSON_PATHS = (
    Path.cwd() / "llm_providers.json",
    Path(__file__).resolve().parent.parent / "llm_providers.json",
)


def _find_json_config(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    for p in _DEFAULT_JSON_PATHS:
        if p.is_file():
            return p
    return None


def _apply_json_overlay(cfg: LLMConfig, json_path: Path) -> None:
    """Merge llm_providers.json values into `cfg` in place.

    Env-supplied fields win: JSON only fills in gaps. This keeps the
    contract clear — the real deployment lever is the environment.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        cfg._source_notes.append(f"json:load-failed:{exc}")
        return

    providers = data.get("providers") or {}
    default_provider = data.get("defaultProvider")

    # Env's LLM_PROVIDER wins if it was set; otherwise use the JSON default.
    provider_name = cfg.provider_name
    env_provider = _env("LLM_PROVIDER")
    if not env_provider and default_provider:
        provider_name = str(default_provider)
        cfg.provider_name = provider_name

    entry = providers.get(provider_name) or {}
    if not entry:
        cfg._source_notes.append(f"json:no-entry-for:{provider_name}")
        return

    # apiKeyEnv points to an env var name; we NEVER put the raw key in JSON.
    if not cfg.api_key:
        key_env = entry.get("apiKeyEnv")
        if key_env:
            cfg.api_key = _env(key_env)
            cfg._source_notes.append(f"json:apiKeyEnv={key_env}")
    if not cfg.api_base and entry.get("apiBase"):
        cfg.api_base = str(entry["apiBase"])
        cfg._source_notes.append("json:apiBase")
    # Only override model if env didn't set one explicitly and JSON has one.
    if _env("LLM_MODEL") is None and entry.get("model"):
        cfg.model = str(entry["model"])
        cfg._source_notes.append("json:model")


def load_config(json_path: Optional[str] = None) -> LLMConfig:
    """Build an :class:`LLMConfig` from env (+ optional JSON overlay).

    Never raises for missing keys — the returned config may have
    ``api_key=None``, in which case ``LLMClient.available`` is False and
    callers must fall back gracefully.
    """
    cfg = LLMConfig()

    # 1. Env pass.
    cfg.provider_name = _env("LLM_PROVIDER") or cfg.provider_name
    cfg.api_key = _env("LLM_API_KEY") or cfg.api_key
    cfg.api_base = _env("LLM_API_BASE") or cfg.api_base
    cfg.model = _env("LLM_MODEL") or cfg.model
    cfg.timeout = _env_float("LLM_TIMEOUT", cfg.timeout) or cfg.timeout
    cfg.temperature = _env_float("LLM_TEMPERATURE", cfg.temperature)
    cfg.max_tokens = _env_int("LLM_MAX_TOKENS", cfg.max_tokens)
    cfg.ca_bundle = _env("LLM_CA_BUNDLE") or cfg.ca_bundle
    cfg.ssl_verify = _env_bool("LLM_SSL_VERIFY", cfg.ssl_verify)
    if cfg.api_key:
        cfg._source_notes.append("env:LLM_API_KEY")

    # 2. Optional JSON overlay (fills gaps only).
    found = _find_json_config(json_path)
    if found is not None:
        cfg._source_notes.append(f"json:{found}")
        _apply_json_overlay(cfg, found)

    # 3. Legacy fallback — the old code used OPENAI_API_KEY / OPENAI_MODEL.
    # Honor them so nothing breaks for existing deployments.
    if not cfg.api_key:
        legacy = _env("OPENAI_API_KEY")
        if legacy:
            cfg.api_key = legacy
            cfg._source_notes.append("env:OPENAI_API_KEY(legacy)")
    if _env("LLM_MODEL") is None and _env("OPENAI_MODEL"):
        cfg.model = _env("OPENAI_MODEL")  # type: ignore[assignment]
        cfg._source_notes.append("env:OPENAI_MODEL(legacy)")

    # 4. CA bundle auto-detect. Only kicks in if the user hasn't set
    # LLM_CA_BUNDLE and SSL_CERT_FILE isn't already pointing somewhere.
    # We prefer the system store because private LLM endpoints may sit
    # behind a CA that certifi doesn't ship.
    if cfg.ca_bundle is None and cfg.ssl_verify:
        env_hint = _env("SSL_CERT_FILE") or _env("REQUESTS_CA_BUNDLE")
        if env_hint and Path(env_hint).is_file():
            cfg.ca_bundle = env_hint
            cfg._source_notes.append("ca:env")
        else:
            detected = _detect_system_ca()
            if detected:
                cfg.ca_bundle = detected
                cfg._source_notes.append(f"ca:auto={detected}")

    return cfg
