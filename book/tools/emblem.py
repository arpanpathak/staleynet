"""Generates the StaleyNET emblem.

    python3 book/tools/emblem.py book/src/figures/emblem.svg

A raised fist grips a rusted chain, a tribute to Layne Staley of Alice in
Chains. The chain runs through the closed hand: it leaves the thumb side
upwards and hangs from the little-finger side, where it ends in the Rust logo,
heavily rusted, for the language the whole system is written in. Where the
links bite into the palm the hand bleeds; the blood pools along the crease,
runs down the wrist and drips. When animated, the hanging chain swings gently
and a drop falls from the wrist.

The hand is drawn the way an illustrator would build it: the back of the hand
in shadow behind four curled fingers (each with a knuckle dome, a wrinkled
middle joint and a tucked fingertip), the thumb across the index and middle
fingers with its nail facing out, and the forearm with two tendons, emerging
from the rolled cuff of a flannel shirt. Every surface is shaded with gradients only, so the emblem renders identically in
browsers, PDF readers and on GitHub.

The Rust logo is the official single-path artwork of the Rust Foundation,
licensed CC BY 4.0 (https://github.com/rust-lang/rust-artwork), modified here
(rust colours and texture).

`emblem()` returns the drawing as SVG elements in a 512 x 512 box, so the
cover generator can place it at any size.
"""
import math
import re
import sys
from pathlib import Path

BOX = 512
CENTRE = BOX / 2

# Colours of the four residual levels, coarse to fine; the cover's background
# spectrogram uses them.
LEVEL_COLOURS = ["#efe6d2", "#e4b27c", "#c8692f", "#8f3618"]

# The Rust logo: its source file, the width of its own viewBox (mm), and the
# offset that centres its path in that box.
RUST_LOGO = Path(__file__).parent / "assets" / "rust-logo-single-path.svg"
LOGO_SOURCE_SIZE = 27.781801
LOGO_SOURCE_OFFSET = (41.583546, -3.0484439)

# The chain: the centres of consecutive links leaving the thumb side, and of
# the links hanging from the little-finger side, then the gear at the end.
UPPER_CHAIN = [(150, 214), (112, 170), (92, 118), (98, 64), (124, 22)]
HANGING_CHAIN = [(364, 236), (404, 276), (420, 326), (420, 376)]
GEAR_CENTRE = (420, 432)
GEAR_DIAMETER = 104

# Face-on links are ovals this wide with this stroke; edge-on links are bars
# this thick. Each link reaches this far past the centres it joins.
RING_WIDTH, RING_STROKE = 34, 15
EDGE_THICKNESS = 15
LINK_OVERLAP = 26

# The hanging chain swings about the point where it leaves the fist.
SWING_DEGREES = 4
SWING_SECONDS = 4.6

# The flannel cuff the forearm emerges from: where its top edge sits, and the
# size of one plaid repeat.
CUFF_TOP = 446
PLAID = 26

# A drop falls from the end of the longest stream every cycle.
DROP_START = (311, 470)
DROP_FALL = 40
DROP_SECONDS = 2.8

# Outline colour, shared by the hand and the metal.
INK = "#1c0d08"

# Skin from the lit surface to the deepest crease, and blood from its bright
# highlight to the dried edge.
SKIN_LIGHT, SKIN_MID, SKIN_DARK, SKIN_SHADOW = "#f1c7a3", "#d9996f", "#a8613f", "#5e2d1c"
BLOOD_LIGHT, BLOOD, BLOOD_DARK = "#d4202a", "#8e0b12", "#4a0408"


def defs(p):
    """Gradients for skin, nail, metal, the gear and blood, and the corrosion
    texture of the metal, with ids made unique by `p`."""
    finger = (f'<linearGradient id="{p}-finger" x1="0" y1="0" x2="1" y2="0">'
              f'<stop offset="0" stop-color="{SKIN_SHADOW}"/><stop offset="0.18" stop-color="{SKIN_DARK}"/>'
              f'<stop offset="0.45" stop-color="{SKIN_LIGHT}"/><stop offset="0.7" stop-color="{SKIN_MID}"/>'
              f'<stop offset="1" stop-color="{SKIN_SHADOW}"/></linearGradient>')
    knuckle = (f'<radialGradient id="{p}-knuckle" cx="0.5" cy="0.25" r="0.6">'
               f'<stop offset="0" stop-color="#fbe0c8" stop-opacity="0.9"/><stop offset="1" stop-color="{SKIN_LIGHT}" stop-opacity="0"/></radialGradient>')
    thumb = (f'<linearGradient id="{p}-thumb" x1="0" y1="0" x2="0" y2="1">'
             f'<stop offset="0" stop-color="{SKIN_LIGHT}"/><stop offset="0.55" stop-color="{SKIN_MID}"/>'
             f'<stop offset="1" stop-color="{SKIN_SHADOW}"/></linearGradient>')
    wrist = (f'<linearGradient id="{p}-wrist" x1="0" y1="0" x2="1" y2="0">'
             f'<stop offset="0" stop-color="{SKIN_SHADOW}"/><stop offset="0.3" stop-color="{SKIN_MID}"/>'
             f'<stop offset="0.55" stop-color="{SKIN_LIGHT}"/><stop offset="0.85" stop-color="{SKIN_DARK}"/>'
             f'<stop offset="1" stop-color="{SKIN_SHADOW}"/></linearGradient>')
    nail = (f'<linearGradient id="{p}-nail" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0" stop-color="#f7e2d6"/><stop offset="1" stop-color="#c9958a"/></linearGradient>')
    metal = (f'<linearGradient id="{p}-metal" x1="0" y1="0" x2="1" y2="1">'
             '<stop offset="0" stop-color="#e3965a"/><stop offset="0.35" stop-color="#a9501f"/>'
             '<stop offset="0.7" stop-color="#5e260d"/><stop offset="1" stop-color="#2a1006"/></linearGradient>')
    gear = (f'<radialGradient id="{p}-gear" cx="0.38" cy="0.32" r="0.8">'
            '<stop offset="0" stop-color="#e08a46"/><stop offset="0.4" stop-color="#a5481a"/>'
            '<stop offset="0.75" stop-color="#5e260d"/><stop offset="1" stop-color="#2a1006"/></radialGradient>')
    blood = (f'<linearGradient id="{p}-blood" x1="0" y1="0" x2="1" y2="0">'
             f'<stop offset="0" stop-color="{BLOOD_DARK}"/><stop offset="0.4" stop-color="{BLOOD_LIGHT}"/>'
             f'<stop offset="1" stop-color="{BLOOD}"/></linearGradient>')
    corrode = (f'<filter id="{p}-corrode" x="-10%" y="-10%" width="120%" height="120%">'
               '<feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed="7" result="noise"/>'
               '<feColorMatrix in="noise" type="matrix" values="0 0 0 0 0.25  0 0 0 0 0.08  0 0 0 0 0.02  0 0 0 1.8 -0.7" result="pits"/>'
               '<feComposite in="pits" in2="SourceAlpha" operator="in" result="crust"/>'
               '<feMerge><feMergeNode in="SourceGraphic"/><feMergeNode in="crust"/></feMerge></filter>')
    plaid = (f'<pattern id="{p}-plaid" width="{PLAID}" height="{PLAID}" patternUnits="userSpaceOnUse">'
             f'<rect width="{PLAID}" height="{PLAID}" fill="#7a1b17"/>'
             f'<rect width="{PLAID * 0.42}" height="{PLAID}" fill="#160807" opacity="0.55"/>'
             f'<rect width="{PLAID}" height="{PLAID * 0.42}" fill="#160807" opacity="0.55"/>'
             f'<rect x="{PLAID * 0.7}" width="1.6" height="{PLAID}" fill="#d9a441" opacity="0.5"/>'
             f'<rect y="{PLAID * 0.7}" width="{PLAID}" height="1.6" fill="#d9a441" opacity="0.5"/></pattern>')
    fold = (f'<linearGradient id="{p}-fold" x1="0" y1="0" x2="1" y2="0">'
            '<stop offset="0" stop-color="#000" stop-opacity="0.55"/><stop offset="0.45" stop-color="#000" stop-opacity="0"/>'
            '<stop offset="1" stop-color="#000" stop-opacity="0.6"/></linearGradient>')
    return f"<defs>{finger}{knuckle}{thumb}{wrist}{nail}{metal}{gear}{blood}{corrode}{plaid}{fold}</defs>"


# ---- chain -------------------------------------------------------------------

def ring_link(cx, cy, length, width, angle, p, stroke=RING_STROKE):
    """A link seen face on: an oval ring along `angle` degrees."""
    rx, ry = length / 2, width / 2
    shape = f'cx="0" cy="0" rx="{rx}" ry="{ry}" fill="none"'
    return (f'<g transform="translate({cx},{cy}) rotate({angle})">'
            f'<ellipse {shape} stroke="{INK}" stroke-width="{stroke + 6}"/>'
            f'<ellipse {shape} stroke="url(#{p}-metal)" stroke-width="{stroke}" filter="url(#{p}-corrode)"/>'
            f'<path d="M {-rx * 0.55} {-ry - stroke * 0.15} Q 0 {-ry - stroke * 0.3} {rx * 0.5} {-ry - stroke * 0.1}" '
            f'stroke="#f2b47d" stroke-width="2.4" fill="none" opacity="0.75" stroke-linecap="round"/></g>')


def edge_link(cx, cy, length, thickness, angle, p):
    """A link seen edge on: a bar with rounded ends."""
    x = -length / 2
    return (f'<g transform="translate({cx},{cy}) rotate({angle})">'
            f'<rect x="{x - 3}" y="{-thickness / 2 - 3}" width="{length + 6}" height="{thickness + 6}" rx="{thickness / 2 + 3}" fill="{INK}"/>'
            f'<rect x="{x}" y="{-thickness / 2}" width="{length}" height="{thickness}" rx="{thickness / 2}" fill="url(#{p}-metal)" filter="url(#{p}-corrode)"/>'
            f'<rect x="{x + 6}" y="{-thickness / 2 + 3}" width="{length * 0.55}" height="2.4" rx="1.2" fill="#f2b47d" opacity="0.7"/></g>')


def chain(points, p):
    """Links between consecutive points, alternating face-on and edge-on, as
    the links of a real chain turn a quarter turn each."""
    parts = []
    for index, ((x0, y0), (x1, y1)) in enumerate(zip(points, points[1:])):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        length = math.hypot(x1 - x0, y1 - y0) + LINK_OVERLAP
        angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        face_on = index % 2 == 0
        parts.append(ring_link(cx, cy, length, RING_WIDTH, angle, p, RING_STROKE) if face_on
                     else edge_link(cx, cy, length, EDGE_THICKNESS, angle, p))
    return "".join(parts)


# ---- hand ---------------------------------------------------------------------

# The four curled fingers seen from the front, index to little finger: left
# edge, width, top of the big knuckle, and the bottom where the fingertip tucks
# into the palm.
FINGERS = [(152, 60, 152, 284), (209, 62, 136, 286), (268, 58, 146, 282), (322, 50, 170, 272)]


def finger_outline(x, w, top, bottom):
    """A knuckle dome on top, a slight waist below the middle joint, and a
    rounded end where the finger curls into the palm."""
    mid = top + (bottom - top) * 0.5
    return (f"M {x + w * 0.06} {top + w * 0.42} "
            f"C {x + w * 0.04} {top - w * 0.02} {x + w * 0.96} {top - w * 0.02} {x + w * 0.94} {top + w * 0.42} "
            f"C {x + w * 1.01} {mid - 12} {x + w * 0.99} {mid + 8} {x + w * 0.93} {mid + 22} "
            f"C {x + w * 0.9} {bottom - 18} {x + w * 0.8} {bottom} {x + w * 0.5} {bottom} "
            f"C {x + w * 0.2} {bottom} {x + w * 0.1} {bottom - 18} {x + w * 0.07} {mid + 22} "
            f"C {x + w * 0.01} {mid + 8} {x - w * 0.01} {mid - 12} {x + w * 0.06} {top + w * 0.42} Z")


def joint_wrinkles(cx, cy, w):
    """The skin folds over the middle joint, which faces the viewer in a fist."""
    folds = []
    for i, (spread, depth, opacity) in enumerate([(0.34, 6, 0.85), (0.27, 9, 0.6), (0.2, 12, 0.45)]):
        y = cy + i * 6
        folds.append(f'<path d="M {cx - w * spread} {y} Q {cx} {y + depth} {cx + w * spread} {y}" '
                     f'stroke="{SKIN_SHADOW}" stroke-width="{2.2 - i * 0.5}" fill="none" stroke-linecap="round" opacity="{opacity}"/>')
    return "".join(folds)


def finger(x, w, top, bottom, p):
    mid = top + (bottom - top) * 0.5
    outline = finger_outline(x, w, top, bottom)
    knuckle = f'<ellipse cx="{x + w * 0.5}" cy="{top + w * 0.3}" rx="{w * 0.3}" ry="{w * 0.26}" fill="url(#{p}-knuckle)"/>'
    middle_joint = f'<ellipse cx="{x + w * 0.5}" cy="{mid - 4}" rx="{w * 0.26}" ry="12" fill="url(#{p}-knuckle)" opacity="0.6"/>'
    tuck_shadow = (f'<path d="M {x + w * 0.14} {bottom - 30} C {x + w * 0.3} {bottom - 8} {x + w * 0.7} {bottom - 8} {x + w * 0.86} {bottom - 30}" '
                   f'stroke="{SKIN_SHADOW}" stroke-width="5" fill="none" opacity="0.35" stroke-linecap="round"/>')
    return (f'<path d="{outline}" fill="url(#{p}-finger)" stroke="{INK}" stroke-width="3" stroke-linejoin="round"/>'
            f"{knuckle}{middle_joint}{joint_wrinkles(x + w * 0.5, mid - 2, w)}{tuck_shadow}")


def back_of_hand(p):
    """The back of the hand, seen between the knuckles: it slopes away from the
    viewer, so it is drawn in shadow behind the fingers."""
    return (f'<path d="M 156 214 C 170 150 340 130 372 196 C 378 214 376 236 368 250 L 160 250 Z" '
            f'fill="{SKIN_DARK}" stroke="{INK}" stroke-width="3"/>'
            f'<path d="M 172 200 C 200 162 320 150 356 196" stroke="{SKIN_SHADOW}" stroke-width="10" fill="none" opacity="0.5"/>')


def wrist(p):
    """The forearm, widening slightly towards the elbow, with two tendons."""
    return (f'<path d="M 176 300 C 182 360 178 430 166 512 L 364 512 C 350 440 344 370 356 296 Z" '
            f'fill="url(#{p}-wrist)" stroke="{INK}" stroke-width="3"/>'
            f'<path d="M 232 330 C 236 390 236 450 230 512" stroke="{SKIN_LIGHT}" stroke-width="7" fill="none" opacity="0.45" stroke-linecap="round"/>'
            f'<path d="M 222 330 C 226 390 226 450 220 512" stroke="{SKIN_DARK}" stroke-width="2.5" fill="none" opacity="0.35"/>'
            f'<path d="M 292 326 C 290 390 292 450 298 512" stroke="{SKIN_LIGHT}" stroke-width="6" fill="none" opacity="0.4" stroke-linecap="round"/>'
            f'<path d="M 302 326 C 300 390 302 450 308 512" stroke="{SKIN_DARK}" stroke-width="2.5" fill="none" opacity="0.3"/>')


def cuff(p):
    """A rolled flannel sleeve around the forearm, red and black plaid, with
    its folds in shadow."""
    outline = (f"M 150 {CUFF_TOP + 6} C 200 {CUFF_TOP - 8} 320 {CUFF_TOP - 8} 380 {CUFF_TOP + 4} "
               f"L 392 {BOX} L 138 {BOX} Z")
    roll = (f"M 150 {CUFF_TOP + 6} C 200 {CUFF_TOP - 8} 320 {CUFF_TOP - 8} 380 {CUFF_TOP + 4} "
            f"L 382 {CUFF_TOP + 22} C 320 {CUFF_TOP + 12} 200 {CUFF_TOP + 12} 148 {CUFF_TOP + 24} Z")
    return (f'<path d="{outline}" fill="url(#{p}-plaid)" stroke="{INK}" stroke-width="3" stroke-linejoin="round"/>'
            f'<path d="{outline}" fill="url(#{p}-fold)"/>'
            f'<path d="{roll}" fill="#000" opacity="0.28"/>'
            f'<path d="M 152 {CUFF_TOP + 24} C 200 {CUFF_TOP + 12} 320 {CUFF_TOP + 12} 382 {CUFF_TOP + 22}" stroke="{INK}" stroke-width="2.5" fill="none"/>'
            f'<path d="M 214 {CUFF_TOP + 30} C 220 {CUFF_TOP + 50} 216 {CUFF_TOP + 60} 212 {BOX}" stroke="#000" stroke-width="3" fill="none" opacity="0.3"/>'
            f'<path d="M 330 {CUFF_TOP + 28} C 334 {CUFF_TOP + 46} 338 {CUFF_TOP + 58} 342 {BOX}" stroke="#000" stroke-width="3" fill="none" opacity="0.3"/>')


def palm(p):
    """The heel of the hand behind the fingers; the chain runs through it."""
    return (f'<path d="M 146 192 C 120 226 118 296 162 326 C 222 356 318 350 364 312 C 384 284 380 230 370 208 Z" '
            f'fill="url(#{p}-wrist)" stroke="{INK}" stroke-width="3"/>')


def thumb(p):
    """The thumb crosses the index and middle fingers below their middle
    joints: a fleshy base on the left, a joint crease, and the nail at the tip."""
    body = ("M 122 222 C 104 262 120 312 170 318 C 210 322 250 314 280 304 "
            "C 304 296 308 266 292 256 C 276 246 250 250 226 252 "
            "C 200 254 180 246 164 230 C 150 216 134 212 122 222 Z")
    nail = ("M 258 262 C 272 254 292 256 296 268 C 299 280 290 292 274 294 "
            "C 260 295 252 288 252 278 C 252 270 253 266 258 262 Z")
    return (f'<path d="{body}" fill="url(#{p}-thumb)" stroke="{INK}" stroke-width="3" stroke-linejoin="round"/>'
            f'<path d="M 140 236 C 160 252 190 262 226 262" stroke="#fbe0c8" stroke-width="6" fill="none" opacity="0.5" stroke-linecap="round"/>'
            f'<path d="{nail}" fill="url(#{p}-nail)" stroke="{SKIN_SHADOW}" stroke-width="2"/>'
            f'<path d="M 262 266 C 272 261 286 262 291 269" stroke="#fff4ec" stroke-width="2" fill="none" opacity="0.75" stroke-linecap="round"/>'
            f'<path d="M 220 258 C 228 272 228 290 220 308" stroke="{SKIN_SHADOW}" stroke-width="2" fill="none" opacity="0.65"/>'
            f'<path d="M 228 262 C 234 274 234 288 228 302" stroke="{SKIN_SHADOW}" stroke-width="1.3" fill="none" opacity="0.45"/>'
            f'<path d="M 170 318 C 210 330 260 322 290 304" stroke="{SKIN_SHADOW}" stroke-width="6" fill="none" opacity="0.3"/>')


# ---- blood ---------------------------------------------------------------------

def run(points, widths, p):
    """A stream of blood along `points`, `widths[i]` wide at each, ending in a drop."""
    left = [(x - w / 2, y) for (x, y), w in zip(points, widths)]
    right = [(x + w / 2, y) for (x, y), w in zip(points, widths)]
    path = f"M {left[0][0]} {left[0][1]} " + " ".join(f"L {x:.1f} {y:.1f}" for x, y in left[1:])
    tip_x, tip_y = points[-1]
    drop = widths[-1] * 1.25
    path += f" C {tip_x - drop} {tip_y + drop * 0.8} {tip_x - drop * 0.6} {tip_y + drop * 2.2} {tip_x} {tip_y + drop * 2.2}"
    path += f" C {tip_x + drop * 0.6} {tip_y + drop * 2.2} {tip_x + drop} {tip_y + drop * 0.8} {right[-1][0]} {right[-1][1]}"
    path += " " + " ".join(f"L {x:.1f} {y:.1f}" for x, y in reversed(right[:-1])) + " Z"
    shine = f'<ellipse cx="{tip_x - drop * 0.3}" cy="{tip_y + drop * 1.2}" rx="{drop * 0.22}" ry="{drop * 0.4}" fill="#ff9c9c" opacity="0.75"/>'
    return f'<path d="{path}" fill="url(#{p}-blood)" stroke="{BLOOD_DARK}" stroke-width="1.2" stroke-linejoin="round"/>{shine}'


def falling_drop(x, y, size, p):
    """A single drop, pointed at the top."""
    return (f'<path d="M {x} {y - size * 1.7} C {x + size * 0.4} {y - size * 0.6} {x + size} {y} {x} {y + size} '
            f'C {x - size} {y} {x - size * 0.4} {y - size * 0.6} {x} {y - size * 1.7} Z" fill="url(#{p}-blood)" stroke="{BLOOD_DARK}" stroke-width="1"/>'
            f'<ellipse cx="{x - size * 0.3}" cy="{y}" rx="{size * 0.2}" ry="{size * 0.35}" fill="#ff9c9c" opacity="0.7"/>')


def blood(p):
    """Blood seeps where the chain bites into the palm under the ring and little
    fingers, pools along the crease, and runs down the wrist."""
    pool = ("M 284 302 C 292 296 300 300 310 296 C 322 290 330 296 340 290 C 350 284 360 280 364 288 "
            "C 368 298 360 306 352 312 C 344 318 338 314 330 320 C 318 328 306 322 296 324 "
            "C 286 326 280 312 284 302 Z")
    streams = (run([(348, 306), (350, 326), (347, 348), (343, 370), (342, 388)], [11, 10, 9, 8, 7], p)
               + run([(308, 318), (305, 346), (303, 376), (305, 404), (309, 430), (311, 452)], [10, 9, 9, 8, 7, 6], p)
               + run([(328, 320), (329, 334), (328, 346)], [8, 7, 6], p))
    on_chain = run([(380, 248), (383, 262), (382, 274)], [7, 5, 4], p)
    return f'<path d="{pool}" fill="url(#{p}-blood)" stroke="{BLOOD_DARK}" stroke-width="1.2" opacity="0.95"/>{streams}{on_chain}'


# ---- the Rust gear -------------------------------------------------------------

def rust_gear(cx, cy, diameter, p):
    """The rusted Rust logo, `diameter` wide, centred on (`cx`, `cy`)."""
    logo = re.search(r' d="([^"]+)"', RUST_LOGO.read_text()).group(1)
    scale = diameter / LOGO_SOURCE_SIZE
    half = LOGO_SOURCE_SIZE / 2
    placement = (f"translate({cx},{cy}) scale({scale:.4f}) translate({-half:.4f},{-half:.4f}) "
                 f"translate({LOGO_SOURCE_OFFSET[0]},{LOGO_SOURCE_OFFSET[1]})")
    return (f'<path transform="{placement}" d="{logo}" fill="url(#{p}-gear)" stroke="{INK}" '
            f'stroke-width="{2.2 / scale:.4f}" stroke-linejoin="round" filter="url(#{p}-corrode)"/>'
            f'<path transform="translate(-1.5,-1.5) {placement}" d="{logo}" fill="none" stroke="#f6c08a" '
            f'stroke-width="{0.7 / scale:.4f}" opacity="0.45"/>')


def swinging(content, animated):
    """`content`, swinging about the point where the chain leaves the fist."""
    if not animated:
        return f"<g>{content}</g>"

    pivot = f"{HANGING_CHAIN[0][0]} {HANGING_CHAIN[0][1]}"
    swing = (f'<animateTransform attributeName="transform" type="rotate" '
             f'values="-{SWING_DEGREES} {pivot};{SWING_DEGREES} {pivot};-{SWING_DEGREES} {pivot}" '
             f'dur="{SWING_SECONDS}s" repeatCount="indefinite"/>')
    return f"<g>{swing}{content}</g>"


def dripping(p, animated):
    """A drop below the longest stream; when animated it falls and fades."""
    x, y = DROP_START
    drop = falling_drop(x, y, 6, p)
    if not animated:
        return drop

    fall = (f'<animateTransform attributeName="transform" type="translate" values="0 0;0 {DROP_FALL}" '
            f'keyTimes="0;1" dur="{DROP_SECONDS}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="1;1;0" keyTimes="0;0.7;1" dur="{DROP_SECONDS}s" repeatCount="indefinite"/>')
    return f"<g>{fall}{drop}</g>"


def emblem(prefix="emblem", animated=True):
    """The emblem as SVG elements in a 512 x 512 box, back to front."""
    p = prefix
    hanging = swinging(chain(HANGING_CHAIN, p) + rust_gear(*GEAR_CENTRE, GEAR_DIAMETER, p), animated)
    fingers = "".join(finger(x, w, top, bottom, p) for x, w, top, bottom in FINGERS)
    hand = f"{wrist(p)}{palm(p)}{chain(UPPER_CHAIN, p)}{back_of_hand(p)}{fingers}{thumb(p)}"
    return f"{defs(p)}<g>{hanging}{hand}{blood(p)}{cuff(p)}{dripping(p, animated)}</g>"


def main(output):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {BOX} {BOX}">'
           f"<title>StaleyNET emblem</title>{emblem()}</svg>")
    Path(output).write_text(svg)


if __name__ == "__main__":
    main(sys.argv[1])
