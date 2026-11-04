"""Writes transcribed lyrics into a StaleyNET cache's manifest.

Reads the JSON lines that transcribe.py wrote and sets the `lyrics` of every
track that has kept lines, in the manifest's format:
`{"text": ..., "timing": {"start": seconds, "end": seconds}}`. Tracks that
already carry lyrics are left alone. The original manifest is kept beside the
new one as manifest.before-lyrics.json, and the codes are not touched, so no
re-tokenization is needed.

Usage: python3 merge.py <cache> <transcripts.jsonl>
"""

import json
import shutil
import sys
from pathlib import Path


def main():
    cache = Path(sys.argv[1])
    transcripts = Path(sys.argv[2])

    manifest_path = cache / "manifest.json"
    backup = cache / "manifest.before-lyrics.json"
    if not backup.exists():
        shutil.copy2(manifest_path, backup)

    lines_by_id = {}
    for line in transcripts.read_text().splitlines():
        if not line.strip():
            continue

        record = json.loads(line)
        if record["lines"]:
            lines_by_id[record["id"]] = record["lines"]

    manifest = json.loads(backup.read_text())
    added = 0
    for entry in manifest["tracks"]:
        track = entry["track"]
        lines = lines_by_id.get(track["id"])
        if not lines or track.get("lyrics"):
            continue

        track["lyrics"] = [{"text": line["text"], "timing": {"start": line["start"], "end": line["end"]}} for line in lines]
        added += 1

    staging = manifest_path.with_suffix(".json.partial")
    staging.write_text(json.dumps(manifest, ensure_ascii=False))
    staging.replace(manifest_path)

    with_lyrics = sum(1 for entry in manifest["tracks"] if entry["track"].get("lyrics"))
    print(f"added lyrics to {added} tracks; {with_lyrics} of {len(manifest['tracks'])} now have lyrics")


if __name__ == "__main__":
    main()
