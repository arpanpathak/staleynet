"""Builds linkedin.svg: the animated cover above a panel of the project's links.

The cover's animations repeat every 2.7 s and 5.4 s, and its chain swings
every 4.2 s; the swing is slowed to 5.4 s here so that the whole image loops
every 5.4 s, which lets render_gif.sh capture one seamless loop.

Usage: python3 make_linkedin.py   (writes linkedin.svg next to this file)
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
COVER = HERE.parent / "cover.svg"

# Canvas: LinkedIn's 4:5 portrait shape.
WIDTH = 1080
HEIGHT = 1350

# The cover is 1200 × 630; scaled to the canvas width.
COVER_HEIGHT = WIDTH * 630 / 1200

# Left margin of the text.
MARGIN = 72

# Top of the first link row, the height of a row, and the distance between rows.
FIRST_ROW = 790
ROW_HEIGHT = 80
ROW_PITCH = 98

SERIF = "Georgia, 'DejaVu Serif', serif"
SANS = "'DejaVu Sans', 'Liberation Sans', sans-serif"
EMOJI = "'Noto Color Emoji', 'Apple Color Emoji', 'Segoe UI Emoji'"

PAPER = "#efe6d2"
MUTED = "#c9b9a0"
RUST = "#e08a4b"

LINKS = [
    ("📄", "Paper", "github.com/arpanpathak/staleynet/tree/main/paper"),
    ("📘", "Book", "arpanpathak.github.io/staleynet"),
    ("🤗", "Checkpoints", "huggingface.co/arpanpathak/staleynet-checkpoints"),
    ("🗂️", "Dataset", "huggingface.co/datasets/arpanpathak/staleynet-data"),
    ("💻", "Code", "github.com/arpanpathak/staleynet"),
]


def cover_body():
    """The cover's drawing without its outer element, looping every 5.4 s."""
    text = COVER.read_text()
    text = re.sub(r"^<svg[^>]*>", "", text, count=1)
    text = re.sub(r"</svg>\s*$", "", text)
    text = re.sub(r"<title>.*?</title>", "", text, count=1)

    return text.replace('dur="4.2s"', 'dur="5.4s"')


def link_row(top, emoji, label, url):
    """One rounded row: an emoji, a small label and the address."""
    return "\n".join([
        f'<rect x="60" y="{top}" width="{WIDTH - 120}" height="{ROW_HEIGHT}" rx="14" fill="{PAPER}" opacity="0.045"/>',
        f'<text x="92" y="{top + 52}" font-family="{EMOJI}" font-size="34">{emoji}</text>',
        f'<text x="156" y="{top + 33}" font-family="{SANS}" font-size="20" font-weight="bold" fill="{RUST}" letter-spacing="1.5">{label.upper()}</text>',
        f'<text x="156" y="{top + 63}" font-family="{SANS}" font-size="25" fill="{PAPER}">{url}</text>',
    ])


def main():
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<title>StaleyNET</title>",
        "<defs>",
        '<linearGradient id="panel" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0c0605"/><stop offset="1" stop-color="#070404"/></linearGradient>',
        '<linearGradient id="fade" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0c0605" stop-opacity="0"/><stop offset="1" stop-color="#0c0605"/></linearGradient>',
        "</defs>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="url(#panel)"/>',
        f'<svg x="0" y="0" width="{WIDTH}" height="{COVER_HEIGHT:.1f}" viewBox="0 0 1200 630" font-family="{SERIF}">{cover_body()}</svg>',
        f'<rect x="0" y="537" width="{WIDTH}" height="{COVER_HEIGHT - 537 + 2:.0f}" fill="url(#fade)"/>',
        f'<rect x="{MARGIN}" y="600" width="{WIDTH - 2 * MARGIN}" height="2" fill="#b7410e" opacity="0.7"/>',
        f'<text x="{MARGIN}" y="676" font-family="{SERIF}" font-size="38" fill="{PAPER}">Text-to-music with singing, built from scratch</text>',
        f'<text x="{MARGIN}" y="724" font-family="{SANS}" font-size="25" fill="{MUTED}">Rust and cuda-oxide GPU kernels · hand-derived gradients · no frameworks</text>',
    ]

    for index, (emoji, label, url) in enumerate(LINKS):
        parts.append(link_row(FIRST_ROW + index * ROW_PITCH, emoji, label, url))

    parts.append("</svg>")
    (HERE / "linkedin.svg").write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
