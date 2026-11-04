"""Extracts the tracks chosen by `select.py` from `fma_large.zip` into one folder
per artist, each clip next to its caption:

    <output>/artist-<artist id>/<track id>.mp3
    <output>/artist-<artist id>/<track id>.txt

`staleynet prepare --folder` treats every subfolder as one split group, so this
layout keeps all of an artist's songs on the same side of the train/validation
split. Reading a member from the archive checks its CRC, so a damaged download
fails here rather than later.

Usage: arrange.py <fma_large.zip> <selection .tsv> <output directory>
"""

import csv
import os
import sys
import zipfile

# Digits in an FMA track id as stored in the archive, and how many of them
# name the folder that holds the clip (fma_large/065/065424.mp3).
TRACK_ID_DIGITS = 6
FOLDER_DIGITS = 3


def member_name(track_id):
    """Path of a track's clip inside fma_large.zip."""
    padded = f"{track_id:0{TRACK_ID_DIGITS}d}"
    return f"fma_large/{padded[:FOLDER_DIGITS]}/{padded}.mp3"


def write_clip(archive, row, output):
    """Copies one clip and writes its caption beside it."""
    track_id = int(row["track_id"])
    folder = os.path.join(output, f"artist-{row['artist_id']}")
    os.makedirs(folder, exist_ok=True)

    stem = os.path.join(folder, f"{track_id:0{TRACK_ID_DIGITS}d}")
    with archive.open(member_name(track_id)) as source, open(f"{stem}.mp3", "wb") as target:
        target.write(source.read())
    with open(f"{stem}.txt", "w") as caption:
        caption.write(row["caption"] + "\n")


def main(archive_path, selection, output):
    rows = list(csv.DictReader(open(selection), delimiter="\t"))

    with zipfile.ZipFile(archive_path) as archive:
        present = set(archive.namelist())
        available = [row for row in rows if member_name(int(row["track_id"])) in present]
        for row in available:
            write_clip(archive, row, output)

    artists = len({row["artist_id"] for row in available})
    print(f"extracted {len(available)} clips into {artists} artist folders; {len(rows) - len(available)} not in the archive")


if __name__ == "__main__":
    main(*sys.argv[1:4])
