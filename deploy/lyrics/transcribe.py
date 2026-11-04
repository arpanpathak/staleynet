"""Transcribes the singing in every clip of a StaleyNET cache with Whisper.

Runs inside the vLLM container on the training machine. Reads the clip
samples that `staleynet vocoder prepare` stored (<cache>/waveforms/<id>.f16,
24 kHz mono, little-endian half floats), resamples them to Whisper's 16 kHz,
and writes one JSON line per clip to the output file:

    {"id": ..., "language": ..., "lines": [{"text": ..., "start": s, "end": s}, ...]}

Lines that look like Whisper's known failure modes on music (low confidence,
repetition loops, stock phrases from subtitle data) are dropped, and a clip
keeps no lines at all when too few words survive. The output is appended to,
so an interrupted run resumes where it stopped.

Usage: python3 transcribe.py <cache> <output.jsonl> [limit]
"""

import json
import os
import re
import sys
import zlib
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams

# Whisper large-v3 (Radford et al., 2022, https://arxiv.org/abs/2212.04356).
MODEL = "openai/whisper-large-v3"

# Sample rate of the stored clips and the rate Whisper expects, in Hz.
CLIP_RATE = 24_000
WHISPER_RATE = 16_000

# Whisper reads at most 30 seconds at once.
WINDOW_SECONDS = 30.0

# Clips sent to the GPU together.
BATCH = 512

# Tokens Whisper may write per clip; 30 seconds of singing needs far fewer,
# and the cap ends the loops it falls into on instrumental passages.
MAX_TOKENS = 200

# Seconds per timestamp token step in Whisper's output.
TIMESTAMP_PATTERN = re.compile(r"<\|(\d+\.\d+)\|>")
LANGUAGE_PATTERN = re.compile(r"<\|([a-z]{2,3})\|>")

# A line is kept only if its mean token log-probability is at least this
# (the threshold Whisper's own decoder uses to reject a transcription).
MIN_MEAN_LOGPROB = -1.0

# Lines whose text compresses better than this are repetition loops
# (Whisper's own threshold is 2.4).
MAX_COMPRESSION_RATIO = 2.4

# A clip keeps its lyrics only with at least this many words in total.
MIN_WORDS = 4

# Stock phrases Whisper invents on music from its subtitle training data.
PHANTOM_PHRASES = (
    "thank you", "thanks for watching", "subscribe", "subtitles", "transcribed by",
    "amara.org", "www.", ".com", "music", "applause", "♪",
)


def resample(samples):
    """24 kHz to 16 kHz by truncating the spectrum (band-limited, no aliasing)."""
    spectrum = np.fft.rfft(samples)
    target_length = int(len(samples) * WHISPER_RATE / CLIP_RATE)
    kept = spectrum[: target_length // 2 + 1]
    return (np.fft.irfft(kept, n=target_length) * target_length / len(samples)).astype(np.float32)


def load_clip(cache, track_id):
    """The first 30 seconds of a clip at 16 kHz, or None when it is missing."""
    path = cache / "waveforms" / f"{track_id}.f16"
    if not path.is_file():
        return None

    samples = np.fromfile(path, dtype="<f2").astype(np.float32)
    samples = samples[: int(WINDOW_SECONDS * CLIP_RATE)]
    return resample(samples)


def compression_ratio(text):
    """Bytes of text over bytes compressed; loops of repeated words score high."""
    data = text.encode("utf-8")
    return len(data) / max(1, len(zlib.compress(data)))


def is_phantom(text):
    """Whether a line looks like one of Whisper's invented stock phrases."""
    lowered = text.lower()
    return any(phrase in lowered for phrase in PHANTOM_PHRASES)


def parse(output):
    """The language and the kept lines of one Whisper output."""
    text = output.outputs[0].text
    token_ids = output.outputs[0].token_ids
    if os.environ.get("WHISPER_DEBUG"):
        print(repr(output.prompt), repr(text[:400]), list(token_ids[:12]), flush=True)
    logprobs = output.outputs[0].logprobs or []

    language_match = LANGUAGE_PATTERN.search(text)
    language = language_match.group(1) if language_match else None

    # Mean log-probability of the whole output, as a per-clip confidence.
    values = [next(iter(step.values())).logprob for step in logprobs if step]
    mean_logprob = sum(values) / len(values) if values else -10.0

    lines = []
    pieces = TIMESTAMP_PATTERN.split(text)
    # split gives: before, t0, text, t1, before-next, t2, text, t3, ...
    stamps_and_text = pieces[1:]
    for index in range(0, len(stamps_and_text) - 2, 2):
        start = float(stamps_and_text[index])
        words = re.sub(r"<\|[^|]*\|>", "", stamps_and_text[index + 1]).strip()
        end = float(stamps_and_text[index + 2])
        if not words or end <= start:
            continue
        if is_phantom(words) or compression_ratio(words) > MAX_COMPRESSION_RATIO:
            continue
        lines.append({"text": words, "start": start, "end": end})

    word_count = sum(len(line["text"].split()) for line in lines)
    confident = mean_logprob >= MIN_MEAN_LOGPROB
    if not confident or word_count < MIN_WORDS:
        lines = []

    return language, mean_logprob, lines, len(token_ids)


def main():
    cache = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None

    manifest = json.loads((cache / "manifest.json").read_text())
    ids = [entry["track"]["id"] for entry in manifest["tracks"]]

    done = set()
    if out_path.is_file():
        done = {json.loads(line)["id"] for line in out_path.read_text().splitlines() if line.strip()}
    pending = [track_id for track_id in ids if track_id not in done]
    if limit is not None:
        pending = pending[:limit]
    print(f"{len(done)} done, {len(pending)} to transcribe", flush=True)

    llm = LLM(model=MODEL, max_model_len=448, max_num_seqs=BATCH, gpu_memory_utilization=0.35, limit_mm_per_prompt={"audio": 1}, enforce_eager=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, skip_special_tokens=False, logprobs=0)

    with out_path.open("a") as out:
        for first in range(0, len(pending), BATCH):
            batch_ids = []
            prompts = []
            for track_id in pending[first : first + BATCH]:
                audio = load_clip(cache, track_id)
                if audio is None:
                    continue
                batch_ids.append(track_id)
                prompts.append({
                    "encoder_prompt": {"prompt": "", "multi_modal_data": {"audio": (audio, WHISPER_RATE)}},
                    "decoder_prompt": "<|startoftranscript|><|en|><|transcribe|><|0.00|>",
                })

            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            for track_id, output in zip(batch_ids, outputs):
                language, mean_logprob, lines, tokens = parse(output)
                record = {"id": track_id, "language": language, "mean_logprob": round(mean_logprob, 3), "lines": lines}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{first + len(batch_ids)} / {len(pending)}", flush=True)


if __name__ == "__main__":
    main()
