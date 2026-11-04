"""Transcribes the singing in every clip of a StaleyNET cache with Whisper.

Runs inside the vLLM container on the training machine. It reads the clip
samples that `staleynet vocoder prepare` stored (`<cache>/waveforms/<id>.f16`,
24 kHz mono, little-endian half floats), resamples them to Whisper's 16 kHz,
and appends one JSON line per clip to the output file:

    {"id": ..., "mean_logprob": ..., "lines": [{"text": ..., "start": s, "end": s}, ...]}

Whisper is told the language (English) and asked for timestamps, so every kept
line carries its start and end in seconds. Lines that look like Whisper's known
failure modes on music are dropped: "♪" for instrumental passages, stock phrases
from its subtitle training data, and repetition loops. A clip keeps no lines
when its output is unconfident or too few words survive. Clips already in the
output file are skipped, so an interrupted run resumes where it stopped.

Usage: python3 transcribe.py <cache> <output.jsonl> [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from vllm import LLM, SamplingParams

# Whisper large-v3 (Radford et al., 2022, https://arxiv.org/abs/2212.04356).
MODEL = "openai/whisper-large-v3"

# Sample rate of the stored clips and the rate Whisper expects, in Hz.
CLIP_RATE = 24_000
WHISPER_RATE = 16_000

# Whisper reads at most 30 seconds at once.
WINDOW_SECONDS = 30.0

# The decoder's first tokens: transcribe English, with timestamps from 0 s.
DECODER_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|0.00|>"

# Clips sent to the engine together.
BATCH = 512

# Tokens Whisper may write per clip. Thirty seconds of singing needs far
# fewer, and the cap ends the loops it falls into on instrumental passages.
MAX_TOKENS = 200

# Longest sequence the engine reserves memory for, in tokens (Whisper's limit).
MAX_MODEL_LENGTH = 448

# Share of the GPU's memory the engine may use; most of it holds the
# cross-attention cache of the clips in flight.
GPU_MEMORY_SHARE = 0.35

# A clip is kept only if its mean token log-probability is at least this,
# the threshold Whisper's own decoder uses to reject a transcription.
MIN_MEAN_LOGPROB = -1.0

# A line whose text compresses better than this is a repetition loop
# (Whisper's own threshold).
MAX_COMPRESSION_RATIO = 2.4

# A clip keeps its lyrics only with at least this many words in total.
MIN_WORDS = 4

# Mean log-probability given to an output with no tokens at all.
EMPTY_LOGPROB = -10.0

# Stock phrases Whisper invents on music, from its subtitle training data.
PHANTOM_PHRASES = (
    "thank you",
    "thanks for watching",
    "subscribe",
    "subtitles",
    "transcribed by",
    "amara.org",
    "www.",
    ".com",
    "music",
    "applause",
    "♪",
)

# A timestamp token such as <|12.34|>, and any special token.
TIMESTAMP = re.compile(r"<\|(\d+\.\d+)\|>")
SPECIAL_TOKEN = re.compile(r"<\|[^|]*\|>")

Audio = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Line:
    """One sung line with its timing, in seconds from the start of the clip."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Clip:
    """The first 30 seconds of one track, at Whisper's sample rate."""

    id: str
    audio: Audio


@dataclass(frozen=True)
class Transcript:
    """What was kept of one clip; `lines` is empty for instrumental or doubtful clips."""

    id: str
    mean_logprob: float
    lines: list[Line]


def resample(samples: Audio) -> Audio:
    """24 kHz to 16 kHz by truncating the spectrum, which cannot alias."""
    target_length = int(len(samples) * WHISPER_RATE / CLIP_RATE)
    spectrum = np.fft.rfft(samples)[: target_length // 2 + 1]
    resampled = np.fft.irfft(spectrum, n=target_length) * target_length / len(samples)

    return resampled.astype(np.float32)


def load_clip(cache: Path, track_id: str) -> Clip | None:
    """The track's clip, or None when the cache has no waveform for it."""
    path = cache / "waveforms" / f"{track_id}.f16"
    if not path.is_file():
        return None

    samples = np.fromfile(path, dtype="<f2").astype(np.float32)
    return Clip(id=track_id, audio=resample(samples[: int(WINDOW_SECONDS * CLIP_RATE)]))


def compression_ratio(text: str) -> float:
    """Bytes of text over bytes compressed; loops of repeated words score high."""
    data = text.encode("utf-8")
    return len(data) / max(1, len(zlib.compress(data)))


def is_kept(line: Line) -> bool:
    """Whether a line looks like singing rather than one of Whisper's failure modes."""
    lowered = line.text.lower()
    is_phantom = any(phrase in lowered for phrase in PHANTOM_PHRASES)
    is_loop = compression_ratio(line.text) > MAX_COMPRESSION_RATIO

    return bool(line.text) and line.end > line.start and not is_phantom and not is_loop


def timed_lines(text: str) -> list[Line]:
    """The lines of Whisper's output, each between two timestamp tokens.

    Splitting `<|0.00|> a<|6.50|><|9.12|> b<|17.30|>` on the timestamps gives
    `["", "0.00", " a", "6.50", "", "9.12", " b", "17.30", ""]`: after the
    leading piece, timestamps and texts alternate.
    """
    pieces = TIMESTAMP.split(text)[1:]
    lines = []

    for index in range(0, len(pieces) - 2, 2):
        words = SPECIAL_TOKEN.sub("", pieces[index + 1]).strip()
        lines.append(Line(text=words, start=float(pieces[index]), end=float(pieces[index + 2])))

    return lines


def mean_logprob(completion: Any) -> float:
    """The mean log-probability of the tokens Whisper chose."""
    steps = completion.logprobs or []
    values = [next(iter(step.values())).logprob for step in steps if step]

    return sum(values) / len(values) if values else EMPTY_LOGPROB


def transcript(track_id: str, completion: Any) -> Transcript:
    """The kept lines of one clip's output, or none when the clip is doubtful."""
    confidence = mean_logprob(completion)
    lines = [line for line in timed_lines(completion.text) if is_kept(line)]

    word_count = sum(len(line.text.split()) for line in lines)
    if confidence < MIN_MEAN_LOGPROB or word_count < MIN_WORDS:
        lines = []

    return Transcript(id=track_id, mean_logprob=round(confidence, 3), lines=lines)


def pending_ids(cache: Path, output: Path, limit: int | None) -> list[str]:
    """Ids of the cache's tracks that the output file does not have yet."""
    manifest = json.loads((cache / "manifest.json").read_text())
    ids = [entry["track"]["id"] for entry in manifest["tracks"]]

    done: set[str] = set()
    if output.is_file():
        done = {json.loads(line)["id"] for line in output.read_text().splitlines() if line.strip()}

    pending = [track_id for track_id in ids if track_id not in done]
    return pending[:limit] if limit is not None else pending


class Whisper:
    """Whisper large-v3 in vLLM, decoding greedily with timestamps."""

    def __init__(self) -> None:
        self.engine = LLM(
            model=MODEL,
            max_model_len=MAX_MODEL_LENGTH,
            max_num_seqs=BATCH,
            gpu_memory_utilization=GPU_MEMORY_SHARE,
            limit_mm_per_prompt={"audio": 1},
            enforce_eager=True,
        )
        self.sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, skip_special_tokens=False, logprobs=0)

    def transcribe(self, clips: Sequence[Clip]) -> list[Any]:
        """One completion per clip, in order."""
        prompts = [
            {
                "encoder_prompt": {"prompt": "", "multi_modal_data": {"audio": (clip.audio, WHISPER_RATE)}},
                "decoder_prompt": DECODER_PROMPT,
            }
            for clip in clips
        ]
        outputs = self.engine.generate(prompts, self.sampling, use_tqdm=False)

        return [output.outputs[0] for output in outputs]


def transcribe_batch(whisper: Whisper, cache: Path, track_ids: Sequence[str]) -> list[Transcript]:
    """Transcripts of the clips among `track_ids` that have a waveform."""
    loaded = (load_clip(cache, track_id) for track_id in track_ids)
    clips = [clip for clip in loaded if clip is not None]

    completions = whisper.transcribe(clips)
    return [transcript(clip.id, completion) for clip, completion in zip(clips, completions)]


def arguments() -> argparse.Namespace:
    """The command line."""
    parser = argparse.ArgumentParser(description="Transcribe the singing in a StaleyNET cache with Whisper.")
    parser.add_argument("cache", type=Path, help="tokenized cache with waveforms")
    parser.add_argument("output", type=Path, help="JSON-lines file to append to")
    parser.add_argument("--limit", type=int, default=None, help="transcribe at most this many clips")

    return parser.parse_args()


def main() -> None:
    args = arguments()
    pending = pending_ids(args.cache, args.output, args.limit)
    print(f"{len(pending)} clips to transcribe", flush=True)

    whisper = Whisper()
    with args.output.open("a") as output:
        for first in range(0, len(pending), BATCH):
            for record in transcribe_batch(whisper, args.cache, pending[first : first + BATCH]):
                output.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

            output.flush()
            print(f"{min(first + BATCH, len(pending))} / {len(pending)}", flush=True)


if __name__ == "__main__":
    main()
