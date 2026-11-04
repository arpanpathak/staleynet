# Training

Training adjusts the network's parameters so that the recorded music becomes
as probable as possible under the model. This chapter describes the quantity
being minimized, how training examples are assembled, the data and how it is
divided between training and evaluation, the optimizer, and the way progress
is saved.

## What is minimized

At every position of a training sequence, the network predicts a probability
for each of the 1 024 possible values of each tokenizer round at the next
position. The prediction is scored by the cross-entropy, which is the
negative logarithm of the probability the network gave to the value that
actually occurs:

\\[
\text{error at one position} = -\log p(\text{actual value}).
\\]

A confident correct prediction (probability close to one) has an error close
to zero, and a confident wrong one has a large error. The training loss is
the average of this error over all positions, computed separately for each
tokenizer round and then averaged over the four rounds so that each round
counts equally. Errors are measured in nats, the unit of the natural
logarithm. For reference, a network that assigns equal probability to all
1 024 values has an error of \\(\ln 1024 \approx 6.93\\) nats, which is the
level any useful model must improve on.

The GPU kernel that computes the error also computes its gradient in the
same pass, because for this loss the gradient has a simple closed form: the
predicted probabilities minus one at the actual value.

## How training examples are assembled

Each training step processes a batch of sequences, for example eight sequences
of 768 positions, of which 128 hold the text and 640 hold about 13.6 seconds of
audio. Each sequence is assembled as follows.

A song is drawn at random, with each dataset's songs weighted as described in
[Lyrics Conditioning](lyrics.md#mixing-small-and-large-datasets). A window of
the song's tokens is then chosen. In 20 % of cases the window begins at the
start of the song, so that the model also learns how pieces open, and
otherwise it begins at a random point. One of the song's captions is chosen,
together with the lyrics sung during the window. In 10 % of cases both are
replaced by empty text, which teaches the model to generate without a prompt
as well. Finally the inputs and the expected next values are written into a
buffer that is reused for every batch.

The random choices for training step number \\(s\\) are derived from the run's
seed and \\(s\\) alone. A run that is stopped and resumed therefore sees exactly
the batches that it would have seen without the interruption.

## The data and its division

The main dataset is the Song Describer Dataset (Manco et al., 2023): 706
Creative Commons songs from the Jamendo platform, each a two-minute excerpt,
described by 1 106 captions written by listeners. It spans pop, rock,
electronic, folk, classical and other styles. Of the 706 songs, 298 have a
caption that mentions singing or a voice (about 9.9 hours), and 131 are
described as instrumental.

To measure whether the model generalizes, about 10 % of the songs are held out
and never used for training. The division is made by artist rather than by
song, so that no artist appears on both sides. Otherwise the model could
appear to generalize while merely recognizing a familiar voice or production
style. The assignment of an artist to one side is determined by a fixed hash
of the artist's identifier, which keeps it identical on every machine; it
leaves 667 songs for training and 39 for evaluation. The lyrics corpus and the
studio recordings are divided in the same way.

Some experiments train on a part of the main dataset selected by the words in
the captions. Words are matched whole, so that "singer" does not match
"singers".

* The **vocal subset** keeps songs whose captions mention a voice (for
  example "vocal", "singer", "sung", "choir" or "rap") and are not described as
  instrumental. It contains 281 training songs (1.57 million steps) and 19
  evaluation songs.
* The **rock subset** keeps songs whose captions mention rock or a related
  style (grunge, alternative, indie, punk, metal, garage, psychedelic,
  stoner, shoegaze, post-rock, progressive) or its typical sound (distorted or
  electric guitar, riffs). It contains 188 training songs, or 6.2 hours. Some
  of these styles are barely present in the dataset, with two songs described
  as grunge, one as psychedelic and none as soft rock, so the model cannot
  learn them as separate styles from this data.

## Overfitting, and how often each song is seen

A model *overfits* when it learns the particular songs it trains on instead of
what they have in common. The loss on the training songs keeps falling, while
the loss on held-out songs, which the model never sees, stops falling. The
difference between the two, the *generalization gap*, measures how much of
what the model has learned applies only to its training songs. A model that has
overfitted generates by recalling: it plays back stretches of a training song,
and it can return the same song whatever the prompt asks for.

How much a model can memorize depends on how often it sees each song. Training
draws every sequence from a song chosen at random, and a song \\(i\\) with
sampling weight \\(w_i\\) is chosen with probability \\(w_i / W\\), where
\\(W = \sum_j w_j\\) is the total weight of all training songs. A run of
\\(S\\) steps with \\(B\\) sequences per step therefore draws song \\(i\\)

\\[
E_i = S \cdot B \cdot \frac{w_i}{W}
\\]

times on average. Each draw is a randomly placed window of the song, so many
draws of one song show the model most of its content many times over.

Weighting a small dataset more heavily makes it count for more in training, and
it also multiplies how often each of its songs is seen. The table gives
\\(E_i\\) for the three configurations used for the 42.9-million-parameter
model of [Experiments](experiments.md#a-larger-model-on-the-finer-tokens-agx-thor).
The training songs are those of the Song Describer Dataset (667), the
JamendoLyrics songs (73), the author's two recorded takes, and, in the later
runs, the 29 314 rock tracks of the Free Music Archive (Defferrard et al., 2017).

| Run | Steps × sequences | Song Describer | JamendoLyrics | author's takes | FMA rock |
|---|---|---|---|---|---|
| `pq-large` | 20 000 × 8 | 163 | 654 | 1 634 | not used |
| `pq-large-rock` (stopped) | 20 000 × 8 | 35 | 105 | 263 | 4.4 |
| `pq-rock-balanced` | 30 000 × 8 | 15.6 | 7.8 | not used | 7.8 |

In `pq-large`, JamendoLyrics was weighted 4-fold and the two takes 10-fold, so
each lyric-bearing song was drawn about 650 times and each take more than
1 600 times. These are also the only songs with lyrics, so whenever the model
was given lyrics during training, the answer was one of 75 songs it had heard
hundreds of times. In informal listening, the model returned the same song for
different prompts and repeated a line of lyrics from its training data.

`pq-rock-balanced` keeps every song within about 16 draws. It does so by adding
the 29 314 rock tracks at half the weight of a Song Describer song, giving
JamendoLyrics the same half weight, and leaving out the two takes, since no
weighting of two recordings would keep them from being memorized. The Song
Describer songs keep twice the weight of the others because their captions are
full sentences written by listeners, while the rock tracks carry only genre
tags such as "rock, punk, garage". The price is that lyrics become rare: 73 of
30 054 songs carry them, so this model will learn little about words. Singing
recognizable words requires many more songs paired with their lyrics.

Two further measures limit the damage of overfitting. A random 10 % of the
sequences are trained without their caption and lyrics, which teaches the
model the unconditional distribution that guidance compares against (see
[Generation](generation.md)). And a run keeps, in `best/`, the checkpoint with
the lowest held-out loss, so a model that begins to overfit late in a run is
not the one that is used.

## The optimizer

The parameters are updated with AdamW (Loshchilov and Hutter, 2019), which
adapts the step size of each parameter to the recent size of its gradients
and shrinks the weights slightly at every step (weight decay, here 0.1), a
form of regularization. Weight decay is not applied to the normalization
gains, the biases or the embedding tables, whose values carry meaning that
shrinking would distort.

The learning rate first rises linearly from zero over a few hundred steps,
which prevents the large, poorly directed updates of the first steps from
damaging the randomly initialized network. It then falls along a cosine curve
to one tenth of its peak by the end of the run, so that the final updates are
small refinements. If the combined size of all gradients exceeds 1.0, they are
scaled down to that size (Pascanu, Mikolov and Bengio, 2013), which protects
training from rare, very large gradients. All arithmetic is carried out in
32-bit floating point. The peak learning rate and the schedule of each run are
given with its results.

## Saving progress

Training on an edge device runs for hours and can be interrupted. The trainer
keeps two checkpoints in each run's directory.

The `latest/` checkpoint holds the weights together with the optimizer's
internal statistics and the step count, and is rewritten every 250 steps by
default. It allows a stopped run to resume with `--resume` exactly where it
was, without restarting the learning-rate schedule or losing the optimizer's
accumulated statistics; at most 250 steps of work can be lost. Its files are
first written under temporary names and then renamed, so an interruption
during saving cannot leave a half-written file in place.

The `best/` checkpoint holds the weights at the point where the error on the
held-out songs was lowest, which is usually the version worth listening to.
When a run resumes, the trainer reads the best held-out error recorded so far
from the run's log, so that `best/` is only replaced by a genuinely better
model.

A new run can also start from the weights of an earlier one with
`--fine-tune-from`. It then begins with a fresh optimizer and schedule, and the
earlier model may lack the lyrics input, which is added as described in
[Lyrics Conditioning](lyrics.md). Every logged event is appended to
`metrics.jsonl` in the run directory.

## A test that the training loop can learn at all

Before any training on real data, a test checks the training machinery as a
whole. A small network is trained repeatedly on one fixed batch, which it
should be able to memorize. Its error must fall from the level of random
guessing (\\(\ln 7 \approx 1.95\\) nats for seven possible values) to below 0.1
within 300 steps. A wrong sign in a gradient, a missing term or a broken
optimizer state prevents this, even when each individual gradient passes the
numerical check described in [The Model](model.md#computing-gradients-by-hand).
