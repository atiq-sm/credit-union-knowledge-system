"""Shared helpers for dev scripts: load snapshots exactly like the harness does."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "dev" / ".cache"
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "submission"))

from run_eval import load_snapshots  # noqa: E402


def snapshots():
    return load_snapshots(ROOT / "snapshots")


def corpus(rebuild: bool = False):
    """(chunks, dates), cached on disk because building takes a few seconds."""
    from corpus import build_corpus

    CACHE.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "submission" / "corpus.py").stat().st_mtime
    path = CACHE / "corpus.pkl"
    if not rebuild and path.exists():
        data = pickle.loads(path.read_bytes())
        if data["src"] == src:
            return data["chunks"], data["dates"]
    chunks, dates = build_corpus(snapshots())
    path.write_bytes(pickle.dumps({"src": src, "chunks": chunks, "dates": dates}))
    return chunks, dates
