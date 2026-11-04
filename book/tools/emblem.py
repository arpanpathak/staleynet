"""Generates the StaleyNET emblem.

    python3 book/tools/emblem.py book/src/figures/emblem.svg

The emblem is a chain link, a tribute to Alice in Chains, that encloses the
model's view of music. Inside the link, 4 rows of tokens stand for the 4
residual codebooks, shaded from bone (coarse) to deep rust (fine). Row `l` is
shifted right by `l` columns, the delay pattern of MusicGen, so the tokens
form a staircase. When animated, the columns appear one after another, the
way the model generates audio one step at a time.

The chain ends in the Rust logo, heavily rusted, hanging from the right-hand
link: the language the whole system is written in. The logo is the official
single-path artwork of the Rust Foundation, licensed CC BY 4.0
(https://github.com/rust-lang/rust-artwork), modified here (rust colours).

`emblem()` returns the drawing as SVG in a 512 x 512 box, so the cover
generator can place it at any size.
"""
import re
import sys
from pathlib import Path

BOX = 512
CENTRE = BOX / 2

# The main link: a stadium (a rectangle with semicircular ends).
LINK_WIDTH, LINK_HEIGHT = 368, 212
LINK_STROKE = 34

# The neighbouring links, standing upright and threaded through the main link's ends.
SIDE_LINK_WIDTH, SIDE_LINK_HEIGHT = 74, 214
SIDE_LINK_STROKE = 26
SIDE_LINK_OFFSET = 184           # horizontal distance of their centres from the middle

# The token staircase: 4 codebooks over 4 frames, delayed into 7 columns.
LEVELS = 4
FRAMES = 4
CELL = 26
GAP = 7
LEVEL_COLOURS = ["#efe6d2", "#e4b27c", "#c8692f", "#8f3618"]   # coarse to fine

# Animation: one column per beat, a pause at the full staircase, then repeat.
BEAT_SECONDS = 0.45
CYCLE_SECONDS = 5.4
DIMMED_OPACITY = 0.12

# The hanging Rust gear: its source file, size and position below the right-hand link.
RUST_LOGO = Path(__file__).parent / "assets" / "rust-logo-single-path.svg"
LOGO_SOURCE_SIZE = 27.781801             # width of the logo's own viewBox, mm
LOGO_SOURCE_OFFSET = (41.583546, -3.0484439)
GEAR_DIAMETER = 104
GEAR_CENTRE = (CENTRE + SIDE_LINK_OFFSET, 462)
HANGER_TOP = CENTRE + SIDE_LINK_HEIGHT / 2 - 10
SWING_DEGREES = 5
SWING_SECONDS = 4.2

INK = "#170a06"
HIGHLIGHT = "#f6c08a"
RUST_STOPS = [("0", "#e08a45"), ("0.45", "#a8471a"), ("1", "#4a1a08")]


def definitions(prefix):
    """Rust gradients, a corrosion texture and the clip path of the lower half,
    with ids made unique by `prefix`."""
    stops = "".join(f'<stop offset="{offset}" stop-color="{colour}"/>' for offset, colour in RUST_STOPS)
    gradient = f'<linearGradient id="{prefix}-rust" x1="0" y1="0" x2="0" y2="1">{stops}</linearGradient>'
    disc = (f'<radialGradient id="{prefix}-gear" cx="0.38" cy="0.32" r="0.8">'
            '<stop offset="0" stop-color="#e08a46"/><stop offset="0.4" stop-color="#a5481a"/>'
            '<stop offset="0.75" stop-color="#5e260d"/><stop offset="1" stop-color="#2a1006"/></radialGradient>')
    corrosion = (f'<filter id="{prefix}-corrode" x="-10%" y="-10%" width="120%" height="120%">'
                 '<feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed="7" result="noise"/>'
                 '<feColorMatrix in="noise" type="matrix" values="0 0 0 0 0.25  0 0 0 0 0.08  0 0 0 0 0.02  0 0 0 1.8 -0.7" result="pits"/>'
                 '<feComposite in="pits" in2="SourceAlpha" operator="in" result="crust"/>'
                 '<feMerge><feMergeNode in="SourceGraphic"/><feMergeNode in="crust"/></feMerge></filter>')
    lower_half = f'<clipPath id="{prefix}-lower"><rect x="0" y="{CENTRE}" width="{BOX}" height="{CENTRE}"/></clipPath>'
    return f"<defs>{gradient}{disc}{corrosion}{lower_half}</defs>"


def rust_gear(prefix, animated):
    """The rusted Rust logo on a small ring link below the right-hand link,
    swinging gently about the top of the ring when animated."""
    logo_path = re.search(r' d="([^"]+)"', RUST_LOGO.read_text()).group(1)
    cx, cy = GEAR_CENTRE
    scale = GEAR_DIAMETER / LOGO_SOURCE_SIZE
    half = LOGO_SOURCE_SIZE / 2
    placement = (f"translate({cx},{cy}) scale({scale:.4f}) translate({-half:.4f},{-half:.4f}) "
                 f"translate({LOGO_SOURCE_OFFSET[0]},{LOGO_SOURCE_OFFSET[1]})")
    pivot = f"{cx} {HANGER_TOP}"
    swing = (f'<animateTransform attributeName="transform" type="rotate" '
             f'values="-{SWING_DEGREES} {pivot};{SWING_DEGREES} {pivot};-{SWING_DEGREES} {pivot}" '
             f'dur="{SWING_SECONDS}s" repeatCount="indefinite"/>') if animated else ""
    ring_centre = (HANGER_TOP + cy - GEAR_DIAMETER / 2) / 2 + 4
    ring = f'cx="{cx}" cy="{ring_centre:.1f}" rx="11" ry="19" fill="none"'
    return (f"<g>{swing}"
            f'<ellipse {ring} stroke="{INK}" stroke-width="15"/>'
            f'<ellipse {ring} stroke="url(#{prefix}-rust)" stroke-width="8"/>'
            f'<path transform="{placement}" d="{logo_path}" fill="url(#{prefix}-gear)" stroke="{INK}" '
            f'stroke-width="{1.6 / scale:.4f}" stroke-linejoin="round" filter="url(#{prefix}-corrode)"/>'
            f'<path transform="translate(-1.5,-1.5) {placement}" d="{logo_path}" fill="none" stroke="{HIGHLIGHT}" '
            f'stroke-width="{0.7 / scale:.4f}" opacity="0.45"/>'
            "</g>")


def link(cx, cy, width, height, stroke, prefix):
    """One chain link seen face on: a dark edge, a rust body and a thin highlight."""
    x, y, radius = cx - width / 2, cy - height / 2, min(width, height) / 2
    shape = f'x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="none"'
    highlight = f'stroke-dasharray="{width * 0.6} {width * 4}" stroke-dashoffset="{-width * 0.2}"'
    return (f'<rect {shape} stroke="{INK}" stroke-width="{stroke + 8}"/>'
            f'<rect {shape} stroke="url(#{prefix}-rust)" stroke-width="{stroke}"/>'
            f'<rect {shape} stroke="{HIGHLIGHT}" stroke-width="2" {highlight} opacity="0.7"/>')


def side_links(prefix):
    """The neighbouring links, as the parts drawn behind and in front of the
    main link. Each is drawn whole behind it and its lower half again in front,
    so the links appear to pass through one another."""
    behind, in_front = [], []
    for side in (-1, 1):
        drawing = link(CENTRE + side * SIDE_LINK_OFFSET, CENTRE, SIDE_LINK_WIDTH, SIDE_LINK_HEIGHT, SIDE_LINK_STROKE, prefix)
        behind.append(drawing)
        in_front.append(f'<g clip-path="url(#{prefix}-lower)">{drawing}</g>')
    return "".join(behind), "".join(in_front)


def reveal(column):
    """An animation that lights a cell when generation reaches its column."""
    start = column * BEAT_SECONDS / CYCLE_SECONDS
    lit = start + 0.04
    return (f'<animate attributeName="opacity" values="{DIMMED_OPACITY};{DIMMED_OPACITY};1;1;{DIMMED_OPACITY}" '
            f'keyTimes="0;{start:.3f};{lit:.3f};0.93;1" dur="{CYCLE_SECONDS}s" repeatCount="indefinite"/>')


def tokens(animated):
    """The delay-pattern staircase: level `l` of frame `f` sits in column `f + l`."""
    columns = FRAMES + LEVELS - 1
    left = CENTRE - (columns * CELL + (columns - 1) * GAP) / 2
    top = CENTRE - (LEVELS * CELL + (LEVELS - 1) * GAP) / 2

    cells = []
    for level in range(LEVELS):
        for frame in range(FRAMES):
            column = frame + level
            x = left + column * (CELL + GAP)
            y = top + level * (CELL + GAP)
            animation = reveal(column) if animated else ""
            cells.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" rx="6" '
                         f'fill="{LEVEL_COLOURS[level]}" stroke="{INK}" stroke-width="3">{animation}</rect>')
    return "".join(cells)


def emblem(prefix="emblem", animated=True):
    """The emblem as SVG elements in a 512 x 512 box."""
    behind, in_front = side_links(prefix)
    main_link = link(CENTRE, CENTRE, LINK_WIDTH, LINK_HEIGHT, LINK_STROKE, prefix)
    return f"{definitions(prefix)}<g>{rust_gear(prefix, animated)}{behind}{main_link}{in_front}{tokens(animated)}</g>"


def main(output):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {BOX} {BOX}">'
           f"<title>StaleyNET emblem</title>{emblem()}</svg>")
    open(output, "w").write(svg)


if __name__ == "__main__":
    main(sys.argv[1])
