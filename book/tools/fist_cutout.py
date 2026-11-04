"""Cuts the raised fist of the emblem out of its photograph.

    python3 book/tools/fist_cutout.py book/tools/assets/fist.webp

The photograph is "Clenched fist.jpg" by Tomas Gunnarsson (Genusfotografen),
licensed CC BY-SA 4.0 (https://commons.wikimedia.org/wiki/File:Clenched_fist.jpg).
It shows a fist against a smooth grey studio background, so the hand can be
separated without a learned model:

1. the background is modelled as a quadratic surface fitted to the photo's
   right, top and lower-right borders (the arm leaves at the left);
2. a pixel belongs to the hand in proportion to how much darker than that
   surface it is, or how much more coloured (the pink thumbnail is light);
3. enclosed holes (the silver ring) are filled, specks are removed, and the
   edge is softened by a pixel so it blends into any background;
4. the grey that the soft edge pixels took from the background is removed
   from their colour (colour decontamination), so no light halo shows on a
   dark page.

The result is rotated a quarter turn so that the fist is raised with the
forearm entering from below, and saved with its transparency. As a derivative
of the photograph, the cut-out is shared under CC BY-SA 4.0 as well.
"""
import json
import sys
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image
from scipy import ndimage

# The photograph on Wikimedia Commons, and the courtesy header its API asks for.
COMMONS_FILE = "Clenched fist.jpg"
USER_AGENT = "staleynet-emblem/1.0 (https://github.com/arpanpathak/staleynet)"

# The original is 5760 x 3840; a third of that is ample for the emblem.
DOWNSCALE = 3

# Width of the border strips that sample the background, in pixels.
BORDER = 40

# A pixel is fully hand when it is this much darker than the background (on a
# 0 to 1 scale), and starts to count at the lower threshold.
DARKNESS_START, DARKNESS_SPAN = 0.06, 0.10

# The same for colour saturation, which catches the light thumbnail.
SATURATION_START, SATURATION_SPAN = 0.10, 0.08


def download():
    """The full-size photograph from Wikimedia Commons."""
    query = urllib.parse.urlencode({
        "action": "query", "format": "json", "titles": f"File:{COMMONS_FILE}", "prop": "imageinfo", "iiprop": "url",
    })
    request = urllib.request.Request(f"https://commons.wikimedia.org/w/api.php?{query}", headers={"User-Agent": USER_AGENT})
    pages = json.load(urllib.request.urlopen(request))["query"]["pages"]
    url = next(iter(pages.values()))["imageinfo"][0]["url"]

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return Image.open(urllib.request.urlopen(request)).convert("RGB")


def background_model(luminance):
    """The grey studio background, as a quadratic surface fitted to the borders."""
    height, width = luminance.shape
    rows, columns = np.mgrid[0:height, 0:width] / max(height, width)

    border = np.zeros_like(luminance, bool)
    border[:BORDER, :] = True
    border[:, -BORDER:] = True
    border[-BORDER:, width // 2:] = True

    terms = np.stack([np.ones_like(rows), columns, rows, columns * columns, rows * rows, columns * rows], -1)
    coefficients, *_ = np.linalg.lstsq(terms[border], luminance[border], rcond=None)
    return terms @ coefficients


def hand_alpha(rgb, background):
    """Opacity of the hand in every pixel of `rgb` (values 0 to 1), given the
    background's luminance."""
    luminance = rgb @ [0.299, 0.587, 0.114]
    saturation = rgb.max(-1) - rgb.min(-1)

    darkness = background - luminance
    alpha = np.clip((darkness - DARKNESS_START) / DARKNESS_SPAN, 0, 1)
    alpha = np.maximum(alpha, np.clip((saturation - SATURATION_START) / SATURATION_SPAN, 0, 1))

    # Keep the largest solid region, with its holes filled and specks removed.
    solid = ndimage.binary_opening(ndimage.binary_fill_holes(alpha > 0.5), iterations=2)
    labels, count = ndimage.label(solid)
    sizes = ndimage.sum(solid, labels, range(1, count + 1))
    solid = labels == 1 + int(np.argmax(sizes))

    core = ndimage.binary_erosion(solid, iterations=3)
    edge = ndimage.binary_dilation(solid, iterations=2)
    alpha = np.where(solid, np.maximum(alpha, core), alpha * edge)
    return ndimage.gaussian_filter(alpha, 0.8)


def decontaminated(rgb, alpha, background):
    """The hand's own colour at its edges. An edge pixel shows
    alpha * hand + (1 - alpha) * background, so the background's share is
    subtracted and the remainder divided by alpha."""
    grey = background[..., None]
    weight = np.maximum(alpha, 0.05)[..., None]

    return np.clip((rgb - (1 - weight) * grey) / weight, 0, 1)


def main(output):
    photo = download()
    photo = photo.resize((photo.width // DOWNSCALE, photo.height // DOWNSCALE), Image.LANCZOS)

    rgb = np.asarray(photo).astype(np.float32) / 255
    background = background_model(rgb @ [0.299, 0.587, 0.114])
    alpha = hand_alpha(rgb, background)
    rgba = np.dstack([decontaminated(rgb, alpha, background), alpha])

    raised = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").rotate(90, expand=True)
    raised.save(output, quality=92, method=6)


if __name__ == "__main__":
    main(sys.argv[1])
