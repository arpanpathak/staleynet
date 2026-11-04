# Training

Training adjusts the network's parameters so that the recorded music becomes
as probable as possible under the model. This chapter describes the quantity
being minimised, how training examples are assembled, the data and how it is
divided between training and evaluation, the optimiser, and the way progress
is saved.

## What is minimised

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

To measure whether the model generalises, about 10 % of the songs are held out
and never used for training. The division is made by artist rather than by
song, so that no artist appears on both sides. Otherwise the model could
appear to generalise while merely recognising a familiar voice or production
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

## The optimiser

The parameters are updated with AdamW (Loshchilov and Hutter, 2019), which
adapts the step size of each parameter to the recent size of its gradients
and shrinks the weights slightly at every step (weight decay, here 0.1), a
form of regularisation. Weight decay is not applied to the normalisation
gains, the biases or the embedding tables, whose values carry meaning that
shrinking would distort.

The learning rate first rises linearly from zero over a few hundred steps,
which prevents the large, poorly directed updates of the first steps from
damaging the randomly initialised network. It then falls along a cosine curve
to one tenth of its peak by the end of the run, so that the final updates are
small refinements. If the combined size of all gradients exceeds 1.0, they are
scaled down to that size (Pascanu, Mikolov and Bengio, 2013), which protects
training from rare, very large gradients. All arithmetic is carried out in
32-bit floating point. The peak learning rate and the schedule of each run are
given with its results.

## Saving progress

Training on an edge device runs for hours and can be interrupted. The trainer
keeps two checkpoints in each run's directory.

The `latest/` checkpoint holds the weights together with the optimiser's
internal statistics and the step count, and is rewritten every 250 steps by
default. It allows a stopped run to resume with `--resume` exactly where it
was, without restarting the learning-rate schedule or losing the optimiser's
accumulated statistics; at most 250 steps of work can be lost. Its files are
first written under temporary names and then renamed, so an interruption
during saving cannot leave a half-written file in place.

The `best/` checkpoint holds the weights at the point where the error on the
held-out songs was lowest, which is usually the version worth listening to.
When a run resumes, the trainer reads the best held-out error recorded so far
from the run's log, so that `best/` is only replaced by a genuinely better
model.

A new run can also start from the weights of an earlier one with
`--fine-tune-from`. It then begins with a fresh optimiser and schedule, and the
earlier model may lack the lyrics input, which is added as described in
[Lyrics Conditioning](lyrics.md). Every logged event is appended to
`metrics.jsonl` in the run directory.

## A test that the training loop can learn at all

Before any training on real data, a test checks the training machinery as a
whole. A small network is trained repeatedly on one fixed batch, which it
should be able to memorise. Its error must fall from the level of random
guessing (\\(\ln 7 \approx 1.95\\) nats for seven possible values) to below 0.1
within 300 steps. A wrong sign in a gradient, a missing term or a broken
optimiser state prevents this, even when each individual gradient passes the
numerical check described in [The Model](model.md#computing-gradients-by-hand).
