"""Generates the StaleyNET emblem.

    python3 book/tools/emblem.py book/src/figures/emblem.svg

The emblem is a chain link, a tribute to Alice in Chains, that encloses the
model's view of music. Inside the link, 4 rows of tokens stand for the 4
residual codebooks, shaded from bone (coarse) to deep rust (fine). Row `l` is
shifted right by `l` columns, the delay pattern of MusicGen, so the tokens
form a staircase. When animated, the columns appear one after another, the
way the model generates audio one step at a time.

A small gear is bolted to the bottom of the right-hand link and drives the
Rust logo, itself a gear with 32 teeth: the language the whole system is
written in. The small gear has 16 teeth, so when animated it turns twice for
each turn of the logo, in the opposite direction, and their teeth stay meshed. The logo is the
official single-path artwork of the Rust Foundation, licensed CC BY 4.0
(https://github.com/rust-lang/rust-artwork), modified here (rust texture).

The metal is a photograph of rusted steel (Poly Haven, CC0; see
assets/SOURCES.md) laid over the shapes as a texture, with shading on top so
that the links read as round bars and the gears as discs.

`emblem()` returns the drawing as SVG in a 512 x 512 box, so the cover
generator can place it at any size.
"""
import base64
import math
import re
import sys
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"

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

# The rust photograph, and the size of one tile of it in the emblem's units.
RUST_TEXTURE = ASSETS / "rust-texture.jpg"
TEXTURE_TILE = 200

# The Rust logo: its source file and the placement of its own viewBox.
RUST_LOGO = ASSETS / "rust-logo-single-path.svg"
LOGO_SOURCE_SIZE = 27.781801             # width of the logo's own viewBox, mm
LOGO_SOURCE_OFFSET = (41.583546, -3.0484439)

# The logo as a gear: 32 teeth, one pointing straight up in the artwork. Its
# tooth roots lie at 91 % of its outer radius (measured from the artwork), so
# its pitch circle, halfway between roots and tips, is at 95.5 % of it.
LOGO_DIAMETER = 96
LOGO_TEETH = 32
LOGO_ROOT_FRACTION = 0.91
LOGO_PITCH_RADIUS = LOGO_DIAMETER / 2 * (1 + LOGO_ROOT_FRACTION) / 2
TOOTH_HEIGHT = LOGO_DIAMETER / 2 * (1 - LOGO_ROOT_FRACTION) / 2   # pitch circle to tip, and to root

# The driving gear: 16 teeth on the logo's tooth pitch, with 4 round holes
# between its hub and its rim, and a bolt through its centre.
DRIVER_TEETH = 16
DRIVER_PITCH_RADIUS = LOGO_PITCH_RADIUS * DRIVER_TEETH / LOGO_TEETH
DRIVER_HOLES = 4
DRIVER_HOLE_RADIUS = 4.2
DRIVER_HOLE_DISTANCE = 12.5
DRIVER_BOLT = 4.5

# The driver is bolted to the bottom bar of the right-hand link; the logo
# meshes directly below it.
DRIVER_CENTRE = (CENTRE + SIDE_LINK_OFFSET, CENTRE + SIDE_LINK_HEIGHT / 2)
LOGO_CENTRE = (DRIVER_CENTRE[0], DRIVER_CENTRE[1] + DRIVER_PITCH_RADIUS + LOGO_PITCH_RADIUS)

# One turn of the logo; the driver turns LOGO_TEETH / DRIVER_TEETH times as often.
LOGO_TURN_SECONDS = 24

INK = "#170a06"
HIGHLIGHT = "#f6c08a"


def definitions(prefix):
    """The rust texture, the shading laid over it, and the clip path of the
    lower half, with ids made unique by `prefix`."""
    photo = base64.b64encode(RUST_TEXTURE.read_bytes()).decode("ascii")
    texture = (f'<pattern id="{prefix}-rust" patternUnits="userSpaceOnUse" width="{TEXTURE_TILE}" height="{TEXTURE_TILE}">'
               f'<image href="data:image/jpeg;base64,{photo}" width="{TEXTURE_TILE}" height="{TEXTURE_TILE}"/></pattern>')

    # Light from above: a round bar is brighter on its upper side and darker below.
    bar_shade = (f'<linearGradient id="{prefix}-bar-shade" x1="0" y1="0" x2="0" y2="1">'
                 '<stop offset="0" stop-color="#ffd9a8" stop-opacity="0.28"/>'
                 '<stop offset="0.45" stop-color="#ffd9a8" stop-opacity="0"/>'
                 '<stop offset="1" stop-color="#0b0402" stop-opacity="0.55"/></linearGradient>')

    # A disc lit from the upper left, darkening toward its rim.
    disc_shade = (f'<radialGradient id="{prefix}-disc-shade" cx="0.38" cy="0.32" r="0.8">'
                  '<stop offset="0" stop-color="#ffd9a8" stop-opacity="0.25"/>'
                  '<stop offset="0.5" stop-color="#ffd9a8" stop-opacity="0"/>'
                  '<stop offset="1" stop-color="#0b0402" stop-opacity="0.6"/></radialGradient>')

    lower_half = f'<clipPath id="{prefix}-lower"><rect x="0" y="{CENTRE}" width="{BOX}" height="{CENTRE}"/></clipPath>'
    return f"<defs>{texture}{bar_shade}{disc_shade}{lower_half}</defs>"


def spin(centre, degrees, seconds):
    """An endless rotation about `centre` by `degrees` every `seconds`."""
    cx, cy = centre
    return (f'<animateTransform attributeName="transform" type="rotate" from="0 {cx} {cy}" '
            f'to="{degrees} {cx} {cy}" dur="{seconds}s" repeatCount="indefinite"/>')


def gear_outline(centre, teeth, pitch_radius, first_tooth_degrees):
    """A spur gear's outline: trapezoidal teeth between the root and tip circles."""
    cx, cy = centre
    tip = pitch_radius + TOOTH_HEIGHT
    root = pitch_radius - TOOTH_HEIGHT
    pitch_angle = 2 * math.pi / teeth

    # Fractions of one tooth pitch: the flank rises over 15 %, the tip spans 20 %.
    corners = [(-0.35, root), (-0.2, tip), (0.0, tip), (0.15, root)]
    points = []
    for tooth in range(teeth):
        centre_angle = math.radians(first_tooth_degrees) + tooth * pitch_angle
        for fraction, radius in corners:
            angle = centre_angle + (fraction + 0.025) * pitch_angle
            points.append(f"{cx + radius * math.cos(angle):.2f},{cy + radius * math.sin(angle):.2f}")

    return "M" + " L".join(points) + " Z"


def circle_path(cx, cy, radius):
    """A circle as a path, so that it can cut a hole in another path."""
    return (f"M{cx + radius:.2f},{cy:.2f} A{radius},{radius} 0 1 0 {cx - radius:.2f},{cy:.2f} "
            f"A{radius},{radius} 0 1 0 {cx + radius:.2f},{cy:.2f} Z")


def driver(prefix, animated):
    """The small gear, bolted to the bottom of the right-hand link."""
    cx, cy = DRIVER_CENTRE

    # A tooth pitch is 360 / 16 = 22.5 degrees; teeth centred at 11.25 + 22.5 k
    # leave a gap pointing straight down, where the logo's upward tooth sits.
    outline = gear_outline(DRIVER_CENTRE, DRIVER_TEETH, DRIVER_PITCH_RADIUS, 360 / DRIVER_TEETH / 2)
    holes = "".join(
        circle_path(
            cx + DRIVER_HOLE_DISTANCE * math.cos(2 * math.pi * hole / DRIVER_HOLES + math.pi / 4),
            cy + DRIVER_HOLE_DISTANCE * math.sin(2 * math.pi * hole / DRIVER_HOLES + math.pi / 4),
            DRIVER_HOLE_RADIUS,
        )
        for hole in range(DRIVER_HOLES)
    )
    gear = f'path d="{outline} {holes}" fill-rule="evenodd"'

    turns = LOGO_TEETH / DRIVER_TEETH
    rotation = spin(DRIVER_CENTRE, 360, LOGO_TURN_SECONDS / turns) if animated else ""
    bolt = f'cx="{cx}" cy="{cy}"'

    return (f"<g>{rotation}"
            f'<{gear} fill="url(#{prefix}-rust)" stroke="{INK}" stroke-width="2.2" stroke-linejoin="round"/>'
            f'<{gear} fill="url(#{prefix}-disc-shade)"/>'
            "</g>"
            f'<circle {bolt} r="{DRIVER_BOLT}" fill="url(#{prefix}-rust)" stroke="{INK}" stroke-width="2"/>'
            f'<circle {bolt} r="{DRIVER_BOLT}" fill="url(#{prefix}-disc-shade)"/>')


def rust_gear(prefix, animated):
    """The Rust logo as a gear, turned by the driver in the opposite direction."""
    logo_path = re.search(r' d="([^"]+)"', RUST_LOGO.read_text()).group(1)
    cx, cy = LOGO_CENTRE
    scale = LOGO_DIAMETER / LOGO_SOURCE_SIZE
    half = LOGO_SOURCE_SIZE / 2
    placement = (f"translate({cx},{cy}) scale({scale:.4f}) translate({-half:.4f},{-half:.4f}) "
                 f"translate({LOGO_SOURCE_OFFSET[0]},{LOGO_SOURCE_OFFSET[1]})")
    rotation = spin(LOGO_CENTRE, -360, LOGO_TURN_SECONDS) if animated else ""
    logo = f'path transform="{placement}" d="{logo_path}"'

    return (f"<g>{rotation}"
            f'<{logo} fill="url(#{prefix}-rust)" stroke="{INK}" stroke-width="{1.6 / scale:.4f}" stroke-linejoin="round"/>'
            f'<{logo} fill="url(#{prefix}-disc-shade)"/>'
            "</g>")


def link(cx, cy, width, height, stroke, prefix):
    """One chain link seen face on: a dark edge, rusted steel, shading and a thin highlight."""
    x, y, radius = cx - width / 2, cy - height / 2, min(width, height) / 2
    shape = f'x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="none"'
    highlight = f'stroke-dasharray="{width * 0.6} {width * 4}" stroke-dashoffset="{-width * 0.2}"'

    return (f'<rect {shape} stroke="{INK}" stroke-width="{stroke + 8}"/>'
            f'<rect {shape} stroke="url(#{prefix}-rust)" stroke-width="{stroke}"/>'
            f'<rect {shape} stroke="url(#{prefix}-bar-shade)" stroke-width="{stroke}"/>'
            f'<rect {shape} stroke="{HIGHLIGHT}" stroke-width="1.5" {highlight} opacity="0.22"/>')


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
    gears = rust_gear(prefix, animated) + driver(prefix, animated)

    return f"{definitions(prefix)}<g>{behind}{main_link}{in_front}{gears}{tokens(animated)}</g>"


def main(output):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {BOX} {BOX}">'
           f"<title>StaleyNET emblem</title>{emblem()}</svg>")
    open(output, "w").write(svg)


if __name__ == "__main__":
    main(sys.argv[1])
