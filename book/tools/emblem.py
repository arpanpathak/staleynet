"""Generates the StaleyNET emblem.

    python3 book/tools/emblem.py book/src/figures/emblem.png

A raised fist grips a rusted chain, a tribute to Layne Staley of Alice in
Chains. The chain runs through the closed hand: it leaves the top of the fist
on the little-finger side, and below it comes out of the gap under the index
finger, where the links bite into the palm and the hand bleeds. From the last
link hangs the Rust logo, rusted through, for the language the whole system is
written in.

The emblem is built from real material rather than drawn:

* the hand is a photograph, "Clenched fist.jpg" by Tomas Gunnarsson
  (Genusfotografen), CC BY-SA 4.0, cut out by `fist_cutout.py`;
* the chain is rendered as three-dimensional geometry (see `shading.py`), each
  link a tube lit from its true surface orientation, coloured with the
  corrosion of "Very rusty chain.jpg" by W.carter, CC0;
* the blood is a painted film raised into a liquid surface and given a sharp
  highlight, which is what makes a bead of liquid read as wet;
* the gear is the official Rust logo of the Rust Foundation, CC BY 4.0
  (https://github.com/rust-lang/rust-artwork), embossed and rusted.

The emblem is therefore shared under CC BY-SA 4.0. The script writes a square
PNG with a transparent background and, next to it, an SVG of the same name
that embeds a smaller copy, for the pages that expect SVG. `emblem()` returns
that picture as SVG elements in a 512 x 512 box, for the cover generator.
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

from shading import VIEW, Layer, chain_along, detail_texture, link_centreline, lit, occlusion, relief_normals, roughen, soft_shadow

ASSETS = Path(__file__).parent / "assets"
FIST = ASSETS / "fist.webp"
RUST_PHOTO = ASSETS / "rust-chain.jpg"
RUST_LOGO = ASSETS / "rust-logo-single-path.svg"
RENDERED = Path(__file__).parents[1] / "src" / "figures" / "emblem.png"

# The working canvas, in pixels of the cut-out photograph (1280 x 1920).
WIDTH, HEIGHT = 1800, 1960

# Edge length of the square PNG, and of the copy embedded in the SVG.
OUTPUT_SIZE = 1024
EMBEDDED_SIZE = 512

# Box of the SVG drawing, kept for the cover generator.
BOX = 512

# Colours of the four residual levels, coarse to fine; the cover's background
# spectrogram uses them.
LEVEL_COLOURS = ["#efe6d2", "#e4b27c", "#c8692f", "#8f3618"]

# One chain link: wire radius, radius of its rounded ends, half the length of
# its straight sides, and the distance between consecutive links.
WIRE, END_RADIUS, HALF_STRAIGHT = 27, 60, 52
LINK = {"wire": WIRE, "end_radius": END_RADIUS, "half_straight": HALF_STRAIGHT,
        "pitch": 2 * (HALF_STRAIGHT + END_RADIUS) - 2 * WIRE + 4}

# The chain above the fist enters behind the knuckles; the chain below leaves
# through the gap under the index finger, whose lower edge is at HOLE_EDGE_Y.
UPPER_CHAIN = [(660, 760), (640, 420), (575, 40)]
LOWER_CHAIN = [(786, 960), (794, 1150), (806, 1330)]
HOLE_EDGE_Y = 1088

# The gear, and the small hole at its top through which a shackle passes, as a
# fraction of the logo's box. The shackle is a small edge-on ring.
GEAR_SIZE = 330
GEAR_HOLE = (0.498, 0.117)
SHACKLE_WIRE, SHACKLE_RADIUS = 8, 17

# Materials: rust, the dark iron scale where it has flaked, and blood.
RUST = np.array([0.60, 0.28, 0.12])
SCALE = np.array([0.24, 0.13, 0.08])
BLOOD = np.array([0.30, 0.012, 0.018])

# Where the corrosion photograph is sampled for the chain and for the gear:
# magnification, then the top-left corner of the crop.
CHAIN_TEXTURE = (1.2, (60, 30))
GEAR_TEXTURE = (1.3, (420, 60))

# The forearm fades out over this many pixels above FOREARM_END.
FOREARM_END, FOREARM_FADE = 1750, 300

# The blood: a pool along the finger's edge, and streams that run down the palm
# (points, then widths in pixels), each ending in a drop.
BLOOD_POOL = [(712, 1084), (740, 1068), (772, 1066), (806, 1072), (842, 1088), (846, 1102),
              (812, 1100), (780, 1096), (750, 1102), (724, 1100)]
BLOOD_STREAMS = [
    ([(722, 1096), (714, 1136), (706, 1186), (709, 1236), (704, 1282), (706, 1318)], [30, 24, 17, 21, 14, 13]),
    ([(842, 1100), (860, 1136), (870, 1180), (874, 1214)], [26, 18, 20, 14]),
    ([(734, 1100), (742, 1124), (738, 1146)], [18, 13, 12]),
    ([(794, 1096), (796, 1118), (793, 1132)], [30, 24, 18]),
]


def load_fist():
    """The cut-out fist on the working canvas, fading out towards the elbow."""
    fist = np.asarray(Image.open(FIST).convert("RGBA")).astype(np.float32) / 255
    canvas = np.zeros((HEIGHT, WIDTH, 4), np.float32)
    canvas[: fist.shape[0], : fist.shape[1]] = fist

    fade = np.clip((FOREARM_END - np.arange(HEIGHT)) / FOREARM_FADE, 0, 1)[:, None]
    canvas[..., 3] *= fade
    return canvas[..., :3], canvas[..., 3]


def corrosion(size, texture, strength):
    """Rust colour with photographed pitting, mixed with larger patches of dark
    scale, as real corrosion is uneven. Also returns the pitting as a bump map."""
    photo = np.asarray(Image.open(RUST_PHOTO).convert("RGB")).astype(np.float32) / 255
    detail = detail_texture(photo, size, *texture)

    noise = np.random.default_rng(1994).random(size).astype(np.float32)
    patches = ndimage.gaussian_filter(noise, 22)
    patches = np.clip((patches - patches.mean()) / (patches.std() + 1e-6) * 0.6 + 0.35, 0, 1)[..., None]

    base = RUST * (1 - patches) + SCALE * patches
    return base * (1 + (detail - 1) * strength), detail.mean(-1)


def render_chain(path, with_gear):
    """The rusted chain along `path`. With `with_gear`, a shackle hangs from
    the last link; returns the colour, the coverage and the gear's centre."""
    layer = Layer(HEIGHT, WIDTH)
    tip = chain_along(path, LINK, layer, 1)

    gear_centre = None
    if with_gear:
        hole = tip + np.array([0.0, 2.6 * SHACKLE_RADIUS, 0.0])
        half = max(np.linalg.norm((hole - tip)[:2]) / 2 - SHACKLE_RADIUS + SHACKLE_WIRE, 1)
        shackle = link_centreline((tip + hole) / 2, np.array([0.0, 1.0, 0.0]), VIEW, half, SHACKLE_RADIUS)
        layer.tube(shackle, SHACKLE_WIRE, 1)
        gear_centre = (int(hole[0] - (GEAR_HOLE[0] - 0.5) * GEAR_SIZE), int(hole[1] - (GEAR_HOLE[1] - 0.5) * GEAR_SIZE))

    albedo, bump = corrosion((HEIGHT, WIDTH), CHAIN_TEXTURE, 0.5)
    roughen(layer, bump, 4.0)

    rgb = lit(layer.normal, albedo, ambient=0.2, diffuse=0.95, specular=0.14, shininess=10)
    rgb *= occlusion(layer, 40, 0.7)[..., None]
    return np.where(layer.coverage[..., None], rgb, 0), layer.coverage.astype(np.float32), gear_centre


def render_gear(centre):
    """The Rust logo, embossed (its edges bevelled over a few pixels) and rusted."""
    png = cairosvg.svg2png(bytestring=RUST_LOGO.read_bytes(), output_width=GEAR_SIZE, output_height=GEAR_SIZE)
    mask = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA")).astype(np.float32)[..., 3] / 255

    bevel = np.clip(ndimage.distance_transform_edt(mask > 0.5) / 7.0, 0, 1) ** 0.6 * 12
    albedo, _ = corrosion(mask.shape, GEAR_TEXTURE, 0.7)
    rgb = lit(relief_normals(bevel, 1.0), albedo, ambient=0.25, diffuse=0.9, specular=0.12, shininess=10)

    canvas_rgb = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    canvas_alpha = np.zeros((HEIGHT, WIDTH), np.float32)
    x0, y0 = centre[0] - GEAR_SIZE // 2, centre[1] - GEAR_SIZE // 2
    canvas_rgb[y0:y0 + GEAR_SIZE, x0:x0 + GEAR_SIZE] = rgb
    canvas_alpha[y0:y0 + GEAR_SIZE, x0:x0 + GEAR_SIZE] = mask
    return canvas_rgb, canvas_alpha


def stroke(mask, points, widths):
    """Paints a stroke along `points` whose width varies as `widths`."""
    for (start, start_width), (end, end_width) in zip(zip(points, widths), zip(points[1:], widths[1:])):
        steps = max(int(np.hypot(end[0] - start[0], end[1] - start[1]) / 2), 1)

        for t in np.linspace(0, 1, steps):
            x, y = start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t
            radius = max(int((start_width + (end_width - start_width) * t) / 2), 1)
            cv2.circle(mask, (int(x), int(y)), radius, 1.0, -1, cv2.LINE_AA)


def render_blood():
    """Blood where the chain bites under the index finger.

    The film is painted as a shape first, then raised into a liquid surface
    whose height swells from its edges, and lit with a broad sheen and a tight
    highlight. Thin blood at the edges is translucent, so the skin shows through."""
    mask = np.zeros((HEIGHT, WIDTH), np.float32)
    cv2.fillPoly(mask, [np.array(BLOOD_POOL, np.int32)], 1.0, cv2.LINE_AA)

    for points, widths in BLOOD_STREAMS:
        stroke(mask, points, widths)
        x, y = points[-1]
        cv2.ellipse(mask, (x, y + 8), (int(widths[-1] * 0.95), int(widths[-1] * 1.25)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    mask = ndimage.gaussian_filter(mask, 1.2)

    inside = ndimage.distance_transform_edt(mask > 0.5)
    height = np.sqrt(np.clip(inside / 9.0, 0, 1)) * 9
    normal = relief_normals(height, 1.5)

    thickness = np.clip(inside / 6.0, 0, 1)[..., None]
    sheen = lit(normal, BLOOD, ambient=0.45, diffuse=0.55, specular=0.1, shininess=14)
    glint = lit(normal, np.zeros(3), ambient=0, diffuse=0, specular=0.85, shininess=140)
    rgb = sheen * (0.65 + 0.35 * thickness) + glint

    alpha = mask * (0.55 + 0.42 * np.clip(inside / 4.0, 0, 1))
    return np.clip(rgb, 0, 1), np.clip(alpha, 0, 1)


def over(below, above):
    """`above` composited over `below`; each is a (colour, coverage) pair."""
    (below_rgb, below_alpha), (rgb, alpha) = below, above
    out_alpha = alpha + below_alpha * (1 - alpha)
    weighted = rgb * alpha[..., None] + below_rgb * (below_alpha * (1 - alpha))[..., None]

    return weighted / np.maximum(out_alpha, 1e-6)[..., None], out_alpha


def framed(rgb, alpha):
    """The smallest square around everything drawn, centred on it, with a
    small margin."""
    rows, columns = np.nonzero(alpha > 0.05)
    side = int(max(rows.max() - rows.min(), columns.max() - columns.min()) * 1.04)
    centre_y, centre_x = (rows.min() + rows.max()) // 2, (columns.min() + columns.max()) // 2

    image = np.pad(np.dstack([rgb, alpha]), ((side, side), (side, side), (0, 0)))
    top, left = centre_y - side // 2 + side, centre_x - side // 2 + side
    return image[top:top + side, left:left + side]


def compose():
    """The emblem as an RGBA image."""
    fist_rgb, fist_alpha = load_fist()

    upper = render_chain(UPPER_CHAIN, with_gear=False)[:2]
    lower_rgb, lower_alpha, gear_centre = render_chain(LOWER_CHAIN, with_gear=True)
    gear = render_gear(gear_centre)
    blood_rgb, blood_alpha = render_blood()

    # Above the gap under the index finger, the lower chain is inside the hand.
    rows = np.arange(HEIGHT)[:, None]
    lower_alpha = lower_alpha * np.where((rows < HOLE_EDGE_Y) & (fist_alpha > 0.5), 0, 1)

    # The lower chain casts a soft shadow on the palm and the forearm.
    shadow = soft_shadow(lower_alpha, (16, 10), 9, 0.55)
    fist_rgb = fist_rgb * (1 - shadow * (fist_alpha > 0))[..., None]

    picture = over(upper, (fist_rgb, fist_alpha))
    picture = over(picture, (blood_rgb, blood_alpha * (fist_alpha > 0.3)))
    picture = over(picture, gear)
    picture = over(picture, (lower_rgb, lower_alpha))

    # Where the chain bites, its top is wet with the same blood.
    wet = blood_alpha * (lower_alpha > 0.5) * (rows < HOLE_EDGE_Y + 50)
    picture = over(picture, (blood_rgb, wet))

    square = framed(np.clip(picture[0], 0, 1), np.clip(picture[1], 0, 1))
    return Image.fromarray((square * 255).astype(np.uint8), "RGBA")


def svg_document(picture):
    """An SVG that shows `picture`, embedded as PNG, in a BOX x BOX view."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {BOX} {BOX}">'
            f"<title>StaleyNET emblem</title>{svg_image(picture)}</svg>")


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
    picture = compose().resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)
    picture.save(output, optimize=True)
    Path(output).with_suffix(".svg").write_text(svg_document(picture))


if __name__ == "__main__":
    main(sys.argv[1])
