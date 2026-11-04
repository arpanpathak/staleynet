"""Generates book/src/figures/cover.svg from the emblem.

    python3 book/tools/cover.py book/src/figures/cover.svg

The cover sets the emblem (see emblem.py), a rusted chain whose central link
holds a staircase of audio tokens, beside the title. Behind the emblem, a faint spectrogram of 4 bands
pulses in the colours of the tokenizer's 4 residual levels, from bone (coarse)
to rust (fine): music, turned into tokens, generated one step at a time.
"""
import math
import sys

from emblem import BOX, LEVEL_COLOURS, emblem

WIDTH, HEIGHT = 1200, 630
EMBLEM_SIZE = 500
EMBLEM_CENTRE = (915, 300)

# The faint spectrogram behind the emblem: bands, bars per band, bar spacing.
BANDS = 4
BARS = 38
BAR_PITCH = 13
BAR_WIDTH = 7
BAND_SPACING = 26
LOWEST_BASELINE = 470
PULSE_SECONDS = 2.7

BACKGROUND_STOPS = [("0", "#1d0f0b"), ("0.55", "#120907"), ("1", "#070404")]
TITLE_COLOUR = "#f1e8d8"
SUBTITLE_COLOUR = "#c9b9a2"
NOTE_COLOUR = "#8f7d6b"
CREDIT_COLOUR = "#5f5248"
SANS = 'font-family="Helvetica, Arial, sans-serif"'


def background():
    """A dark radial gradient, lightest behind the emblem."""
    stops = "".join(f'<stop offset="{offset}" stop-color="{colour}"/>' for offset, colour in BACKGROUND_STOPS)
    return (f'<defs><radialGradient id="cover-background" cx="0.72" cy="0.5" r="0.85">{stops}</radialGradient></defs>'
            f'<rect width="{WIDTH}" height="{HEIGHT}" fill="url(#cover-background)"/>')


def pulsing_bar(x, baseline, height, colour, delay):
    """One bar that grows to 1.8 times its height and back."""
    peak = height * 1.8
    timing = f'dur="{PULSE_SECONDS}s" begin="{delay:.2f}s" repeatCount="indefinite"'
    return (f'<rect x="{x:.1f}" y="{baseline - height:.1f}" width="{BAR_WIDTH}" height="{height:.1f}" rx="2" fill="{colour}" opacity="0.10">'
            f'<animate attributeName="height" values="{height:.1f};{peak:.1f};{height:.1f}" {timing}/>'
            f'<animate attributeName="y" values="{baseline - height:.1f};{baseline - peak:.1f};{baseline - height:.1f}" {timing}/>'
            f"</rect>")


def spectrogram():
    """4 rows of faint bars, one per token level, rising and falling in waves."""
    left = EMBLEM_CENTRE[0] - BARS * BAR_PITCH / 2
    bars = []
    for band in range(BANDS):
        baseline = LOWEST_BASELINE - band * BAND_SPACING
        for index in range(BARS):
            height = 6 + 10 * (1 + math.sin(index * 0.55 + band * 1.3))
            delay = index * 0.06 + band * 0.25
            bars.append(pulsing_bar(left + index * BAR_PITCH, baseline, height, LEVEL_COLOURS[band], delay))
    return "".join(bars)


def placed_emblem():
    """The emblem scaled to EMBLEM_SIZE and centred on EMBLEM_CENTRE."""
    scale = EMBLEM_SIZE / BOX
    x = EMBLEM_CENTRE[0] - EMBLEM_SIZE / 2
    y = EMBLEM_CENTRE[1] - EMBLEM_SIZE / 2
    return f'<g transform="translate({x:.1f},{y:.1f}) scale({scale:.4f})">{emblem("cover")}</g>'


def title():
    """Name, tagline, method line and dedication."""
    return (
        f'<text x="80" y="250" font-size="92" font-weight="700" letter-spacing="1" fill="{TITLE_COLOUR}">StaleyNET</text>'
        f'<rect x="84" y="276" width="96" height="4" rx="2" fill="{LEVEL_COLOURS[2]}"/>'
        f'<text x="82" y="330" font-size="28" font-style="italic" fill="{SUBTITLE_COLOUR}">Music from a sentence.</text>'
        f'<text x="84" y="374" font-size="17" letter-spacing="1.6" fill="{NOTE_COLOUR}" {SANS}>RESIDUAL TOKENS · DELAY PATTERN · TRANSFORMER</text>'
        f'<text x="84" y="400" font-size="17" letter-spacing="1.6" fill="{NOTE_COLOUR}" {SANS}>RUST AND CUDA-OXIDE, TRAINED FROM SCRATCH</text>'
        f'<text x="84" y="560" font-size="17" font-style="italic" fill="{NOTE_COLOUR}">For Layne Staley (1967 to 2002) and Alice in Chains.</text>'
        f'<text x="84" y="590" font-size="11" fill="{CREDIT_COLOUR}" {SANS}>Rust texture: W.carter, CC0. Rust logo by the Rust Foundation, CC BY 4.0, modified. No endorsement implied.</text>'
    )


def main(output):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
           f'font-family="Georgia, \'Times New Roman\', serif"><title>StaleyNET</title>'
           f"{background()}{spectrogram()}{placed_emblem()}{title()}</svg>")
    open(output, "w").write(svg)


if __name__ == "__main__":
    main(sys.argv[1])
