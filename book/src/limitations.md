# Limitations and Future Work

This chapter collects the known weaknesses of the system, beginning with those
that most affect what a listener hears, and outlines the work that would
address them.

## Limits on what the output sounds like

The most important limitation is the fidelity of the audio representation.
The experiments in
[Section 5 of Experiments](experiments.md#5-the-audio-pipeline-in-isolation)
show that the tokenizer loses about 0.7 log units of spectral detail, roughly
3 dB per frequency band, and that a model which reproduces its training audio
perfectly still sounds only as good as this reconstruction. The detail that
singing depends on, the harmonics and formants that make vowels
recognizable, is precisely what is lost. Until the representation improves,
longer training of the language model cannot make voices intelligible.

Griffin-Lim adds a further loss that the spectral measure understates: it
estimates phase without ever having heard real audio, so harmonics lose their
frame-to-frame coherence and voices take on a metallic smear (see
[Audio Representation](audio.md#why-griffin-lim-smears-voices)).
[The neural vocoder](vocoder.md), trained on spectrograms decoded from
StaleyNET's own tokens, is meant to replace it. It is trained with spectral
losses only, and vocoders trained that way tend to sound slightly muffled in
the highest frequencies; adversarial training, as in Vocos, would address
this.

The lyrics input does not yet influence the output measurably. The first
models trained with it produced the same predictions with correct lyrics as
with lyrics from another song. Learning which sounds belong to which words
from 78 songs in four languages may require more lyric-bearing data than is
available, or a stronger form of conditioning.

The training data contains comparatively little singing. Of the 706 songs in
the main dataset, 298 mention a voice in their captions, and several styles
of interest are nearly absent: two songs are described as grunge, one as
psychedelic, and none as soft rock.

Small datasets invite overfitting, and the first 42.9-million-parameter model
overfitted: its training loss kept falling while its held-out loss stopped at
5.06 nats per code, and it returned nearly the same song for different prompts
([Experiments](experiments.md#the-model-overfitted)). The cause was heavy
weighting of the smallest datasets, under which some songs were drawn more
than a thousand times. The balanced run that replaced it draws no song more
than about 16 times, but it pays for this by making lyric-bearing songs rare.
The model also has no dropout inside the network; the only regularization is
weight decay, prompt dropout, and keeping the checkpoint with the lowest
held-out loss.

The scale of the system is small by the standards of published music
generators. Its largest model has 42.9 million parameters and is trained on
about 300 hours of audio, most of it rock tracks described only by genre tags,
whereas MusicGen trains between 300 million and 3.3 billion parameters on
20 000 hours. The output is mono at 24 kHz.

## Limits of the implementation

All arithmetic is carried out in 32-bit floating point, and the GPU's tensor
cores, which accelerate lower-precision matrix products, are not used. Storing
values in 16-bit form while accumulating in 32 bits would roughly halve the
memory traffic.

Attention stores the full matrix of scores between all pairs of positions,
which limits the context length that fits in memory. A fused attention kernel
in the style of FlashAttention (Dao et al., 2022) would remove that matrix.

At every training step the loss of each tokenizer round and the size of every
gradient are copied from the GPU to the CPU, which forces the CPU to wait for
the GPU and limits the speed-up on the faster AGX Thor.

The spectrogram is computed as a matrix product, which costs about a hundred
times more arithmetic than an FFT for the frame length used, and is therefore
only 1.3 times faster than the CPU on the Orin NX.

## Future work

The analysis above suggests an order of priorities.

1. **A better audio representation.** A learned tokenizer, consisting of a
   convolutional encoder and decoder trained together with residual
   quantization, as in SoundStream, would preserve far more detail at a
   similar number of tokens. A lighter intermediate step is to quantize
   single frames instead of pairs, or to use more and larger codebooks, and to
   measure the effect with `staleynet reconstruct`.
2. **Adversarial training for the vocoder**: discriminators that judge short
   stretches of waveform and of spectrogram, added to the spectral losses
   once those have converged.
3. **More lyric-bearing singing data**, in particular short takes recorded with
   the studio, in which each line can be aligned with the audio.
4. **Cross-attention to the lyrics**, so that every audio position can consult
   the words directly rather than through the shared sequence.
5. **Ablation studies** of the guidance scale, the rate at which text is
   removed during training, and the delay pattern.
6. **Engineering improvements**: mixed precision, tensor-core matrix
   products, fused attention, and fewer synchronisations between GPU and CPU.
