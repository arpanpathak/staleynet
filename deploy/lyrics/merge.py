"""Writes transcribed lyrics into a StaleyNET cache's manifest.

Reads the JSON lines that transcribe.py wrote and sets the `lyrics` of every
track that has kept lines, in the manifest's format:

    {"text": ..., "timing": {"start": seconds, "end": seconds}}

Tracks that already carry lyrics are left alone. The original manifest is kept
beside the new one as `manifest.before-lyrics.json` and is always the starting
point, so merging twice gives the same result. The codes are not touched, so
nothing needs to be tokenized again.

Usage: python3 merge.py <cache> <transcripts.jsonl>
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, TypedDict

# The manifest, and the copy of it taken before the first merge.
MANIFEST = "manifest.json"
ORIGINAL = "manifest.before-lyrics.json"


class Timing(TypedDict):
    """When a line is sung, in seconds from the start of the clip."""

    start: float
    end: float


class LyricLine(TypedDict):
    """One line in the manifest's lyrics format."""

    text: str
    timing: Timing


def read_lyrics(transcripts: Path) -> dict[str, list[LyricLine]]:
    """The kept lines of every transcribed clip, by track id."""
    lyrics: dict[str, list[LyricLine]] = {}

    for row in transcripts.read_text().splitlines():
        if not row.strip():
            continue

        record = json.loads(row)
        lines = [LyricLine(text=line["text"], timing=Timing(start=line["start"], end=line["end"])) for line in record["lines"]]
        if lines:
            lyrics[record["id"]] = lines

    return lyrics


def original_manifest(cache: Path) -> dict[str, Any]:
    """The manifest as it was before any lyrics were merged into it."""
    original = cache / ORIGINAL
    if not original.exists():
        shutil.copy2(cache / MANIFEST, original)

    manifest: dict[str, Any] = json.loads(original.read_text())
    return manifest


def add_lyrics(manifest: dict[str, Any], lyrics: dict[str, list[LyricLine]]) -> int:
    """Gives each track without lyrics its transcribed lines; returns how many tracks gained some."""
    added = 0

    for entry in manifest["tracks"]:
        track = entry["track"]
        lines = lyrics.get(track["id"])
        if not lines or track.get("lyrics"):
            continue

        track["lyrics"] = lines
        added += 1

    return added


def write_atomically(path: Path, manifest: dict[str, Any]) -> None:
    """Writes `manifest` under a temporary name and renames it into place."""
    staging = path.with_suffix(".json.partial")
    staging.write_text(json.dumps(manifest, ensure_ascii=False))
    staging.replace(path)


def arguments() -> argparse.Namespace:
    """The command line."""
    parser = argparse.ArgumentParser(description="Merge transcribed lyrics into a StaleyNET cache.")
    parser.add_argument("cache", type=Path, help="cache whose manifest receives the lyrics")
    parser.add_argument("transcripts", type=Path, help="JSON lines written by transcribe.py")

    return parser.parse_args()


def main() -> None:
    args = arguments()
    manifest = original_manifest(args.cache)
    added = add_lyrics(manifest, read_lyrics(args.transcripts))
    write_atomically(args.cache / MANIFEST, manifest)

    with_lyrics = sum(1 for entry in manifest["tracks"] if entry["track"].get("lyrics"))
    print(f"added lyrics to {added} tracks; {with_lyrics} of {len(manifest['tracks'])} now have lyrics")


if __name__ == "__main__":
    main()
