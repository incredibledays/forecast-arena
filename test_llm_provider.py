"""Smoke test for the LLM layer.

Loads config from env / llm_providers.json, prints a masked summary,
then sends a tiny prompt and echoes the response. Exits 0 if the
layer works AND is unconfigured (so CI stays green when no key is
present); exits non-zero only on a genuine parse/import bug.

Usage:
    python test_llm_provider.py
    python test_llm_provider.py --json      # exercise chat_json too
"""

from __future__ import annotations

import argparse
import json
import sys

from llm import LLMClient, load_config, mask_key
from llm.client import _extract_json_substring, FALLBACK_FORECAST


def _print_header(cfg) -> None:
    print("=" * 60)
    print("ForecastArena — LLM provider smoke test")
    print("=" * 60)
    print(f"provider_name : {cfg.provider_name}")
    print(f"model         : {cfg.model}")
    print(f"api_base      : {cfg.api_base or '<default>'}")
    print(f"api_key       : {mask_key(cfg.api_key)}")
    print(f"timeout       : {cfg.timeout}")
    print(f"temperature   : {cfg.temperature}")
    print(f"max_tokens    : {cfg.max_tokens}")
    print(f"source_notes  : {cfg._source_notes}")
    print("-" * 60)


def _self_test_json_extract() -> bool:
    """Run the extractor over known-tricky inputs. No network needed."""
    cases = [
        ('{"a":1}', {"a": 1}),
        ('prefix {"a":1} suffix', {"a": 1}),
        ('```json\n{"a":1,"b":[2,3]}\n```', {"a": 1, "b": [2, 3]}),
        ('nested {"a":{"b":2}} trailing', {"a": {"b": 2}}),
    ]
    ok = True
    for raw, want in cases:
        got = _extract_json_substring(raw)
        try:
            parsed = json.loads(got) if got else None
        except Exception:
            parsed = None
        status = "PASS" if parsed == want else "FAIL"
        if parsed != want:
            ok = False
        print(f"  [{status}] extract({raw!r:40s}) -> {parsed}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true",
                    help="Also test chat_json with a forecast-shaped prompt.")
    args = ap.parse_args()

    cfg = load_config()
    _print_header(cfg)

    # Local extractor test — always runs, no network.
    print("JSON extractor self-test:")
    extractor_ok = _self_test_json_extract()
    print("-" * 60)

    client = LLMClient(cfg)
    if not client.available:
        print(f"LLM client not available: {client.unavailable_reason}")
        print("(This is fine — the app still runs; agents fall back to heuristics.)")
        return 0 if extractor_ok else 1

    # --- live call: plain chat ---
    print("Sending plain chat prompt...")
    try:
        reply = client.chat(
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Say 'ready' in one word."},
            ],
            max_tokens=16,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"chat() raised: {exc}")
        return 2
    print(f"chat reply    : {reply!r}")
    print("-" * 60)

    # --- live call: JSON chat ---
    if args.json:
        print("Sending chat_json prompt...")
        try:
            payload = client.chat_json(
                messages=[
                    {"role": "system", "content":
                     "Return ONLY a JSON object matching the requested schema."},
                    {"role": "user", "content":
                     'Return: {"probability_yes": 0.5, "confidence": 0.5, '
                     '"reasoning_summary": "test", "key_evidence": [], '
                     '"risk_factors": []}'},
                ],
                max_tokens=200,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"chat_json() raised: {exc}")
            return 3
        print(f"chat_json     : {json.dumps(payload, indent=2)}")
        if payload == FALLBACK_FORECAST:
            print("(Fallback used — provider returned unparseable content.)")
        print("-" * 60)

    print("OK.")
    return 0 if extractor_ok else 1


if __name__ == "__main__":
    sys.exit(main())
