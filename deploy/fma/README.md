# The FMA rock corpus

These scripts add the rock family of the Free Music Archive (FMA) to the
training data: 32,923 Creative Commons clips of 30 seconds, about 274 hours,
against the 23 hours of the Song Describer Dataset.

| Step | Command |
|---|---|
| Download | `fma_metadata.zip` (342 MB) and `fma_large.zip` (93 GB) from <https://github.com/mdeff/fma> |
| Select | `python3 deploy/fma/select.py data/raw/fma/fma_metadata data/raw/fma/rock.tsv` |
| Everything after | `deploy/fma/pipeline.sh`: extract, compute spectrograms, tokenize, start `pq-xl` |

**Selection.** A track is kept when any of its genres lies under Rock in the
FMA genre tree. The largest subgenres are punk (9,261 tracks), lo-fi, indie
rock, garage, psych-rock, loud-rock, industrial, noise-rock, post-rock and
post-punk; shoegaze, sludge and the metal styles are smaller.

**Captions.** FMA has no written descriptions, so each caption is assembled
from metadata: the track's rock subgenres, up to two other genres, up to four
listener tags, and the tempo and energy measured by Echo Nest where available,
for example `rock, post-punk, punk, synth pop, pop, 184 bpm, medium energy`.
These captions are shorter and more formulaic than the Song Describer ones,
which is why `configs/pq-xl-thor.json` keeps the human-written captions at
about 14 % of the training draws.

**Split.** `arrange.py` places each artist in a folder of its own, and the
folder catalog treats a folder as one split group, so no artist appears in both
the training and the validation songs.

**Licenses.** Every clip carries the Creative Commons license chosen by its
artist (recorded in `rock.tsv`); about half are NonCommercial, and some are
NoDerivatives. The derived caches are research artifacts under the same terms
as the rest of the training data.
