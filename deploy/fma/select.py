"""Selects the rock-family tracks of the Free Music Archive and writes a caption
for each one.

The FMA (Defferrard et al., 2017, <https://arxiv.org/abs/1612.01840>) tags every
track with genres from a tree whose top level includes Rock. A track is kept
when any of its genres lies under Rock. Its caption joins, in order:

* its rock subgenres, or "rock" when it carries only the top-level genre;
* up to two genres from outside the rock family, such as "noise" or "pop";
* up to four listener tags that are short plain words;
* the tempo in beats per minute and a coarse energy level, for the tracks
  that Echo Nest analysed.

The captions are therefore generated from metadata, not written by listeners.

Usage: select.py <fma_metadata directory> <output .tsv>
"""

import ast
import collections
import csv
import sys

# Subsets of the FMA, from smallest to largest; `fma_large` holds all three.
SUBSETS = ("small", "medium", "large")

# Length of every clip in `fma_large`, in seconds.
CLIP_SECONDS = 30

# Genres from outside the rock family added to a caption.
OTHER_GENRES_KEPT = 2

# Listener tags added to a caption, and the longest tag accepted.
TAGS_KEPT = 4
LONGEST_TAG = 24

# Echo Nest energy (0 to 1) above which a track is "high energy", and below
# which it is "calm".
HIGH_ENERGY = 0.7
LOW_ENERGY = 0.3


def load_genres(metadata):
    """Genre rows by id, and a function that returns a genre's top-level ancestor."""
    genres = {row["genre_id"]: row for row in csv.DictReader(open(f"{metadata}/genres.csv"))}

    def top_level(genre_id):
        while genres[genre_id]["parent"] != "0":
            genre_id = genres[genre_id]["parent"]
        return genre_id

    return genres, top_level


def load_echonest(metadata):
    """Tempo (beats per minute) and energy (0 to 1) by track id."""
    rows = list(csv.reader(open(f"{metadata}/echonest.csv")))
    header = [f"{a}.{b}.{c}" for a, b, c in zip(rows[0], rows[1], rows[2])]
    tempo_column = header.index("echonest.audio_features.tempo")
    energy_column = header.index("echonest.audio_features.energy")

    return {row[0]: (float(row[tempo_column]), float(row[energy_column])) for row in rows[4:]}


def plain_tags(tags):
    """Up to TAGS_KEPT listener tags that are short words, lowercased, without repeats."""
    kept = []
    for tag in ast.literal_eval(tags or "[]"):
        tag = tag.strip().lower()
        is_plain = all(character.isalnum() or character in " -'&" for character in tag)
        if tag and len(tag) <= LONGEST_TAG and is_plain and tag not in kept:
            kept.append(tag)
    return kept[:TAGS_KEPT]


def energy_word(energy):
    """A coarse description of an Echo Nest energy value."""
    if energy > HIGH_ENERGY:
        return "high energy"
    if energy < LOW_ENERGY:
        return "calm"
    return "medium energy"


def caption(rock_names, other_names, tags, analysis):
    """The caption of one track, as described in the module documentation."""
    parts = (rock_names or ["rock"]) + other_names + [tag for tag in tags if tag not in rock_names]
    if analysis is not None:
        tempo, energy = analysis
        parts += [f"{round(tempo)} bpm", energy_word(energy)]

    names_mention_rock = "rock" in " ".join(rock_names)
    ordered = parts if names_mention_rock else ["rock"] + parts
    return ", ".join(dict.fromkeys(ordered))


def main(metadata, output):
    genres, top_level = load_genres(metadata)
    rock = next(genre["genre_id"] for genre in genres.values() if genre["title"] == "Rock")
    echonest = load_echonest(metadata)

    rows = list(csv.reader(open(f"{metadata}/tracks.csv")))
    column = {f"{a}.{b}": index for index, (a, b) in enumerate(zip(rows[0], rows[1]))}

    selected, subgenre_counts = [], collections.Counter()
    for row in rows[3:]:
        track_id = row[0]
        genre_ids = [str(genre) for genre in ast.literal_eval(row[column["track.genres_all"]] or "[]")]
        if row[column["set.subset"]] not in SUBSETS or not any(top_level(genre) == rock for genre in genre_ids):
            continue

        rock_names = [genres[g]["title"].lower() for g in genre_ids if top_level(g) == rock and genres[g]["title"] != "Rock"]
        other_names = [genres[g]["title"].lower() for g in genre_ids if top_level(g) != rock][:OTHER_GENRES_KEPT]
        text = caption(rock_names, other_names, plain_tags(row[column["track.tags"]]), echonest.get(track_id))

        selected.append((track_id, row[column["artist.id"]], text, row[column["track.license"]]))
        subgenre_counts.update(rock_names or ["rock"])

    with open(output, "w") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["track_id", "artist_id", "caption", "license"])
        writer.writerows(selected)

    print(len(selected), "tracks,", round(len(selected) * CLIP_SECONDS / 3600, 1), "hours of", CLIP_SECONDS, "s clips")
    print(subgenre_counts.most_common())


if __name__ == "__main__":
    main(*sys.argv[1:3])
