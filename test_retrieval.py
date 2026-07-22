"""Temporary CLI smoke test for the Tavily retrieval layer.

Usage:
    python test_retrieval.py "OpenAI IPO 2026"

Prints up to 5 normalized results (title / url / summary / score).
If TAVILY_API_KEY is missing, prints a warning and exits 0 with an
empty result set — the rest of the app must keep working.
"""

import argparse
import sys

from retrieval import TavilyProvider


def main():
    parser = argparse.ArgumentParser(description="Smoke test Tavily retrieval.")
    parser.add_argument("query", nargs="+", help="search query")
    parser.add_argument("--max-results", type=int, default=5)
    args = parser.parse_args()

    query = " ".join(args.query).strip()
    if not query:
        print("empty query", file=sys.stderr)
        sys.exit(2)

    provider = TavilyProvider()
    print(f"query: {query!r}")
    print(f"provider enabled: {provider.enabled}")

    results = provider.search(query, max_results=args.max_results)
    print(f"results: {len(results)}")
    print()

    if not results:
        print("(no results — check TAVILY_API_KEY and network)")
        return

    for i, r in enumerate(results, start=1):
        print(f"--- [{i}] score={r['relevance_score']:.3f}")
        print(f"title: {r['title']}")
        print(f"url  : {r['url']}")
        summary = r["content_summary"]
        if len(summary) > 300:
            summary = summary[:300].rstrip() + "…"
        print(f"summary: {summary}")
        print()


if __name__ == "__main__":
    main()
