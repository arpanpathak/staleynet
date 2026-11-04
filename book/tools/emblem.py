"""Generates the StaleyNET emblem.

    python3 book/tools/emblem.py book/src/figures/emblem.png

A rusted chain, a tribute to Layne Staley of Alice in Chains. Its large
central link encloses the model's view of music: 4 rows of enamelled tiles, one
per residual level of the tokenizer from bone (coarse) to deep rust (fine),
each row shifted one column to the right as in the delay pattern, so that the
tiles form a staircase. The chain continues to either side, and from the
central link's lower wire hangs the Rust logo on a small shackle, rusted
through: the language the whole system is written in.

Nothing is drawn by hand. The chain is three-dimensional geometry rendered by
`shading.py`: every link is a tube swept along a stadium-shaped centreline,
consecutive links turn a quarter turn as in a real chain, a depth buffer lets
them interlock, and every pixel is lit from its true surface orientation, with
ambient occlusion in the crevices. The rust colour and pitting come from
"Very rusty chain.jpg" by W.carter (CC0), with the photograph's own lighting
divided out. The tiles are embossed: a bevelled height field lit with a
glossy highlight, a dark rim where the enamel thins over the edge, and a soft
shadow. The gear is the official Rust logo of the Rust Foundation, CC BY 4.0
(https://github.com/rust-lang/rust-artwork), embossed and rusted.

The script writes a square PNG with a transparent background and, next to it,
an SVG of the same name that embeds a smaller copy, for pages that expect SVG.
`emblem()` returns that picture as SVG elements in a 512 x 512 box, for the
cover generator.
"""
import base64
import io
import sys
from pathlib import Path

import cairosvg
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from shading import VIEW, Layer, detail_texture, link_centreline, lit, occlusion, relief_normals, roughen, soft_shadow

ASSETS = Path(__file__).parent / "assets"
RUST_PHOTO = ASSETS / "rust-chain.jpg"
RUST_LOGO = ASSETS / "rust-logo-single-path.svg"
RENDERED = Path(__file__).parents[1] / "src" / "figures" / "emblem.png"

# The working canvas, the edge of the finished square, and the copy embedded
# in the SVG, in pixels.
CANVAS = 1400
OUTPUT_SIZE = 1024
EMBEDDED_SIZE = 512

# Box of the SVG drawing, kept for the cover generator.
BOX = 512

# Margin around the drawing in the finished square, as a fraction of its size.
MARGIN = 0.04

# The central link: centre, half the length of its straight sides, the radius
# of its rounded ends (to the wire's centre), and the wire's radius.
CENTRE = (700, 560)
MAIN_HALF_STRAIGHT, MAIN_END_RADIUS, MAIN_WIRE = 168, 150, 34

# The links on either side, and how many of them there are per side.
SIDE_HALF_STRAIGHT, SIDE_END_RADIUS, SIDE_WIRE = 34, 42, 26
SIDE_LINKS = 2

# The gear, the small hole at its top as a fraction of the logo's box, and the
# shackle that joins that hole to the central link's lower wire.
GEAR_SIZE = 300
GEAR_HOLE = (0.498, 0.117)
SHACKLE_WIRE, SHACKLE_RADIUS = 10, 20

# Rust, and the dark iron scale that shows where it has flaked.
RUST = np.array([0.60, 0.28, 0.12])
SCALE = np.array([0.24, 0.13, 0.08])

# Where the corrosion photograph is sampled: magnification, then the top-left
# corner of the crop.
CHAIN_TEXTURE = (1.0, (60, 30))
GEAR_TEXTURE = (1.3, (420, 60))
TILE_TEXTURE = (2.2, (100, 500))

# The token staircase: residual levels, frames, and the size, gap and corner
# radius of each tile. Level colours run from bone (coarse) to deep rust (fine);
# the cover's background spectrogram uses them too.
LEVEL_COLOURS = ["#efe6d2", "#e4b27c", "#c8692f", "#8f3618"]
LEVELS, FRAMES = 4, 4
CELL, GAP, CORNER = 46, 10, 9


def corrosion(size, texture, strength, seed=1994):
    """Rust colour with photographed pitting, mixed with larger patches of dark
    scale, as real corrosion is uneven. Also returns the pitting as a bump map."""
    photo = np.asarray(Image.open(RUST_PHOTO).convert("RGB")).astype(np.float32) / 255
    detail = detail_texture(photo, size, *texture)

    noise = np.random.default_rng(seed).random(size).astype(np.float32)
    patches = ndimage.gaussian_filter(noise, 18)
    patches = np.clip((patches - patches.mean()) / (patches.std() + 1e-6) * 0.6 + 0.35, 0, 1)[..., None]

    base = RUST * (1 - patches) + SCALE * patches
    return base * (1 + (detail - 1) * strength), detail.mean(-1)


def horizontal_link(layer, x, half_straight, end_radius, wire, face_on):
    """One link with its long axis horizontal, centred at (x, CENTRE[1]); seen
    face on, or edge on (turned a quarter turn about its axis)."""
    axis = np.array([1.0, 0.0, 0.0])
    side = np.array([0.0, 1.0, 0.0]) if face_on else VIEW
    centreline = link_centreline((x, CENTRE[1], 0), axis, side, half_straight, end_radius)

    layer.tube(centreline, wire, 1)


def side_links(layer):
    """The links that hook the central link's ends, alternating edge on and
    face on outwards. Hooked links overlap by two wire radii."""
    main_reach = MAIN_HALF_STRAIGHT + MAIN_END_RADIUS
    side_reach = SIDE_HALF_STRAIGHT + SIDE_END_RADIUS

    for direction in (-1, 1):
        x = CENTRE[0] + direction * (main_reach + side_reach - 2 * SIDE_WIRE + 2)
        for index in range(SIDE_LINKS):
            horizontal_link(layer, x, SIDE_HALF_STRAIGHT, SIDE_END_RADIUS, SIDE_WIRE, face_on=index % 2 == 1)
            x += direction * (2 * side_reach - 2 * SIDE_WIRE + 2)


def shackle(layer):
    """A small edge-on ring around the central link's lower wire; returns
    where the gear's centre must be for its top hole to hang on the ring."""
    bottom = CENTRE[1] + MAIN_END_RADIUS
    hole = bottom + 3.2 * SHACKLE_RADIUS
    half = (hole - bottom) / 2 - SHACKLE_RADIUS + SHACKLE_WIRE

    ring = link_centreline((CENTRE[0], (bottom + hole) / 2, 0), np.array([0.0, 1.0, 0.0]), VIEW, half, SHACKLE_RADIUS)
    layer.tube(ring, SHACKLE_WIRE, 1)

    return int(CENTRE[0] - (GEAR_HOLE[0] - 0.5) * GEAR_SIZE), int(hole - (GEAR_HOLE[1] - 0.5) * GEAR_SIZE)


def render_chain():
    """The rusted chain: colour, coverage, and the centre of the hanging gear."""
    layer = Layer(CANVAS, CANVAS)
    horizontal_link(layer, CENTRE[0], MAIN_HALF_STRAIGHT, MAIN_END_RADIUS, MAIN_WIRE, face_on=True)
    side_links(layer)
    gear_centre = shackle(layer)

    albedo, bump = corrosion((CANVAS, CANVAS), CHAIN_TEXTURE, 0.5)
    roughen(layer, bump, 4.0)

    rgb = lit(layer.normal, albedo, ambient=0.2, diffuse=0.95, specular=0.16, shininess=10)
    rgb *= occlusion(layer, 30, 0.7)[..., None]
    return np.where(layer.coverage[..., None], rgb, 0), layer.coverage.astype(np.float32), gear_centre


def render_gear(centre):
    """The Rust logo, embossed (its edges bevelled over a few pixels) and rusted."""
    png = cairosvg.svg2png(bytestring=RUST_LOGO.read_bytes(), output_width=GEAR_SIZE, output_height=GEAR_SIZE)
    mask = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA")).astype(np.float32)[..., 3] / 255

    bevel = np.clip(ndimage.distance_transform_edt(mask > 0.5) / 7.0, 0, 1) ** 0.6 * 12
    albedo, _ = corrosion(mask.shape, GEAR_TEXTURE, 0.7)
    rgb = lit(relief_normals(bevel, 1.0), albedo, ambient=0.25, diffuse=0.9, specular=0.12, shininess=10)

    canvas_rgb = np.zeros((CANVAS, CANVAS, 3), np.float32)
    canvas_alpha = np.zeros((CANVAS, CANVAS), np.float32)
    x0, y0 = centre[0] - GEAR_SIZE // 2, centre[1] - GEAR_SIZE // 2
    canvas_rgb[y0:y0 + GEAR_SIZE, x0:x0 + GEAR_SIZE] = rgb
    canvas_alpha[y0:y0 + GEAR_SIZE, x0:x0 + GEAR_SIZE] = mask
    return canvas_rgb, canvas_alpha


def rounded_square(mask, x, y):
    """Paints one CELL x CELL tile with corners rounded by CORNER into `mask`."""
    cv2.rectangle(mask, (x + CORNER, y), (x + CELL - CORNER, y + CELL), 1, -1)
    cv2.rectangle(mask, (x, y + CORNER), (x + CELL, y + CELL - CORNER), 1, -1)

    for corner_x in (x + CORNER, x + CELL - CORNER):
        for corner_y in (y + CORNER, y + CELL - CORNER):
            cv2.circle(mask, (corner_x, corner_y), CORNER, 1, -1)


def render_tiles():
    """The delay-pattern staircase: tile (level, frame) sits in column
    frame + level, so each finer level runs one step behind the coarser one."""
    columns = FRAMES + LEVELS - 1
    left = CENTRE[0] - (columns * CELL + (columns - 1) * GAP) / 2
    top = CENTRE[1] - (LEVELS * CELL + (LEVELS - 1) * GAP) / 2

    mask = np.zeros((CANVAS, CANVAS), np.uint8)
    colour = np.zeros((CANVAS, CANVAS, 3), np.float32)
    for level in range(LEVELS):
        glaze = np.array([int(LEVEL_COLOURS[level][i:i + 2], 16) for i in (1, 3, 5)]) / 255

        for frame in range(FRAMES):
            tile = np.zeros_like(mask)
            rounded_square(tile, int(left + (frame + level) * (CELL + GAP)), int(top + level * (CELL + GAP)))
            mask |= tile
            colour[tile > 0] = glaze

    coverage = ndimage.gaussian_filter(mask.astype(np.float32), 0.6)
    inside = ndimage.distance_transform_edt(coverage > 0.5)
    normal = relief_normals(np.clip(inside / 7.0, 0, 1) ** 0.7 * 8, 0.8)

    _, grain = corrosion((CANVAS, CANVAS), TILE_TEXTURE, 0.0, seed=7)
    worn = 1 + (grain - 1) * 0.07
    rgb = lit(normal, colour * worn[..., None], ambient=0.42, diffuse=0.62, specular=0.38, shininess=36)

    # A dark rim, as enamel thins over a stamped edge.
    rim = np.clip(1 - inside / 2.5, 0, 1)[..., None]
    return np.clip(rgb * (1 - 0.55 * rim), 0, 1), coverage


def over(below, above):
    """`above` composited over `below`; each is a (colour, coverage) pair."""
    (below_rgb, below_alpha), (rgb, alpha) = below, above
    out_alpha = alpha + below_alpha * (1 - alpha)
    weighted = rgb * alpha[..., None] + below_rgb * (below_alpha * (1 - alpha))[..., None]

    return weighted / np.maximum(out_alpha, 1e-6)[..., None], out_alpha


def framed(rgba):
    """The smallest square around everything drawn, centred on it, with MARGIN."""
    rows, columns = np.nonzero(rgba[..., 3] > 0.05)
    side = int(max(rows.max() - rows.min(), columns.max() - columns.min()) * (1 + MARGIN))
    centre_y, centre_x = (rows.min() + rows.max()) // 2, (columns.min() + columns.max()) // 2

    padded = np.pad(rgba, ((side, side), (side, side), (0, 0)))
    top, left = centre_y - side // 2 + side, centre_x - side // 2 + side
    return padded[top:top + side, left:left + side]


def compose():
    """The emblem as a square RGBA image."""
    chain_rgb, chain_alpha, gear_centre = render_chain()
    gear = render_gear(gear_centre)
    tile_rgb, tile_alpha = render_tiles()
    tile_shadow = soft_shadow(tile_alpha, (6, 4), 4, 0.45)

    empty = np.zeros((CANVAS, CANVAS, 3), np.float32)
    picture = over((empty, np.zeros((CANVAS, CANVAS), np.float32)), gear)
    picture = over(picture, (chain_rgb, chain_alpha))
    picture = over(picture, (empty, tile_shadow))
    picture = over(picture, (tile_rgb, tile_alpha))

    square = framed(np.dstack([np.clip(picture[0], 0, 1), np.clip(picture[1], 0, 1)]))
    image = Image.fromarray((square * 255).astype(np.uint8), "RGBA")
    return image.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)


def svg_image(picture):
    """`picture` as an SVG image element filling the BOX x BOX box."""
    buffer = io.BytesIO()
    picture.resize((EMBEDDED_SIZE, EMBEDDED_SIZE), Image.LANCZOS).save(buffer, "PNG", optimize=True)
    data = base64.b64encode(buffer.getvalue()).decode()

    return f'<image width="{BOX}" height="{BOX}" href="data:image/png;base64,{data}"/>'


def emblem(prefix="emblem", animated=True):
    """The rendered emblem as SVG elements in a BOX x BOX box. The emblem is a
    picture, so `prefix` and `animated` are accepted for compatibility only."""
    return svg_image(Image.open(RENDERED))


def main(output):
    picture = compose()
    picture.save(output, optimize=True)

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {BOX} {BOX}">'
           f"<title>StaleyNET emblem</title>{svg_image(picture)}</svg>")
    Path(output).with_suffix(".svg").write_text(svg)


if __name__ == "__main__":
    main(sys.argv[1])
