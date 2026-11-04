"""Physically shaded tubes for the emblem: chain links and their lighting.

A tube is a centreline in three dimensions swept by a sphere of radius r.
Splatting spheres along the centreline into a depth buffer gives, for every
covered pixel, the depth and the normal of the visible surface. Interlocking
links therefore hide each other correctly, and every pixel can be lit from its
true orientation with the Lambert and Blinn-Phong models
(https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model).

Coordinates are image pixels: x to the right, y downwards, z towards the viewer.
"""
import numpy as np
from scipy import ndimage

# Unit vector towards the light: upper left and slightly in front, matching the
# studio lighting of the hand photograph.
LIGHT = np.array([-0.45, -0.65, 0.62]) / np.linalg.norm([-0.45, -0.65, 0.62])

# Unit vector towards the viewer, and the half vector between it and the light.
VIEW = np.array([0.0, 0.0, 1.0])
HALF = (LIGHT + VIEW) / np.linalg.norm(LIGHT + VIEW)

# Sphere spacing along a tube, as a fraction of its radius; smaller is smoother.
SPHERE_SPACING = 0.12

# Points sampled along the centreline of one link.
LINK_SAMPLES = 160


class Layer:
    """The visible surface of a group of tubes: depth, normal and material per pixel."""

    def __init__(self, height, width):
        self.depth = np.full((height, width), -np.inf, np.float32)
        self.normal = np.zeros((height, width, 3), np.float32)
        self.material = np.zeros((height, width), np.int16)

    @property
    def coverage(self):
        """True where any tube is visible."""
        return np.isfinite(self.depth)

    def sphere(self, centre, radius, material):
        """Splats one sphere, keeping whichever surface is nearest the viewer."""
        x, y, z = centre
        height, width = self.depth.shape
        x0, x1 = max(int(x - radius - 1), 0), min(int(x + radius + 2), width)
        y0, y1 = max(int(y - radius - 1), 0), min(int(y + radius + 2), height)
        if x0 >= x1 or y0 >= y1:
            return

        rows, columns = np.mgrid[y0:y1, x0:x1]
        dx, dy = (columns - x) / radius, (rows - y) / radius
        inside = dx * dx + dy * dy < 1
        dz = np.sqrt(np.clip(1 - dx * dx - dy * dy, 0, 1))
        surface = z + dz * radius

        region = self.depth[y0:y1, x0:x1]
        nearer = inside & (surface > region)
        region[nearer] = surface[nearer]
        self.normal[y0:y1, x0:x1][nearer] = np.stack([dx, dy, dz], -1)[nearer]
        self.material[y0:y1, x0:x1][nearer] = material

    def tube(self, points, radius, material):
        """Sweeps a sphere of `radius` along the 3D polyline `points`."""
        points = np.asarray(points, np.float32)
        step = max(radius * SPHERE_SPACING, 0.5)

        for start, end in zip(points, points[1:]):
            count = max(int(np.linalg.norm(end[:2] - start[:2]) / step), 1)
            for t in np.linspace(0, 1, count, endpoint=False):
                self.sphere(start + (end - start) * t, radius, material)

        self.sphere(points[-1], radius, material)


def link_centreline(centre, axis, side, half_straight, end_radius):
    """The centreline of one stadium-shaped link: two straight sides joined by
    semicircles. `axis` runs along the chain and `side` across the link."""
    centre, axis, side = (np.asarray(vector, np.float32) for vector in (centre, axis, side))
    quarter = LINK_SAMPLES // 4

    points = []
    for sign in (1, -1):
        for along in np.linspace(-half_straight, half_straight, quarter) * sign:
            points.append(centre + axis * along + side * end_radius * sign)

        tip = centre - axis * half_straight * sign
        for angle in np.linspace(0, np.pi, quarter):
            points.append(tip + side * end_radius * sign * np.cos(angle) - axis * sign * end_radius * np.sin(angle))

    points.append(points[0])
    return np.array(points)


def chain_along(path, link, layer, material):
    """Interlocking links along the 2D polyline `path`, alternating face on and
    edge on as the links of a real chain turn a quarter turn each. `link` holds
    `pitch`, `half_straight`, `end_radius` and `wire`. Returns the far tip of
    the last link's centreline."""
    path = np.asarray(path, np.float32)
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(segments)])

    tip = None
    for index in range(int(arc[-1] // link["pitch"]) + 1):
        distance = index * link["pitch"]
        k = min(np.searchsorted(arc, distance, side="right") - 1, len(segments) - 1)
        direction = (path[k + 1] - path[k]) / max(segments[k], 1e-6)
        point = path[k] + direction * (distance - arc[k])

        axis = np.array([direction[0], direction[1], 0.0])
        across = np.array([-direction[1], direction[0], 0.0])
        side = across if index % 2 == 0 else VIEW
        centre = np.array([point[0], point[1], 0.0])

        layer.tube(link_centreline(centre, axis, side, link["half_straight"], link["end_radius"]), link["wire"], material)
        tip = centre + axis * (link["half_straight"] + link["end_radius"])

    return tip


def lit(normal, albedo, ambient, diffuse, specular, shininess):
    """Colour of a surface with `normal` and `albedo` under LIGHT."""
    lambert = np.clip(normal @ LIGHT, 0, 1)[..., None]
    highlight = np.clip(normal @ HALF, 0, 1)[..., None] ** shininess

    return albedo * (ambient + diffuse * lambert) + specular * highlight


def relief_normals(height, blur):
    """Normals of the height field `height` (pixels), smoothed by `blur`."""
    slope_y, slope_x = np.gradient(ndimage.gaussian_filter(height, blur))
    normal = np.dstack([-slope_x, -slope_y, np.ones_like(slope_x)])

    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)


def roughen(layer, bump, amount):
    """Tilts the layer's normals by the slope of `bump`, so corrosion pits catch the light."""
    slope_y, slope_x = np.gradient(ndimage.gaussian_filter(bump, 1.2))

    normal = layer.normal.copy()
    normal[..., 0] -= slope_x * amount
    normal[..., 1] -= slope_y * amount
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6

    layer.normal = np.where(layer.coverage[..., None], normal, layer.normal)


def occlusion(layer, radius, strength):
    """Darkening where a nearby surface stands in front of this one: the
    crevices between interlocking links receive less light."""
    depth = np.where(layer.coverage, layer.depth, np.nan)
    floor = np.nanmin(depth) if layer.coverage.any() else 0.0
    filled = np.where(layer.coverage, depth, floor)

    in_front = np.clip((ndimage.maximum_filter(filled, size=int(radius)) - filled) / radius, 0, 1)
    return 1 - strength * ndimage.gaussian_filter(in_front, radius / 4)


def detail_texture(photo, size, scale, offset):
    """Corrosion detail from a photograph: the photo divided by a blurred copy
    of itself, so its own lighting is removed and only the pitting and the
    colour remain. `scale` magnifies the crop that starts at `offset`."""
    height, width = size
    crop = photo[offset[1]:offset[1] + int(height / scale) + 2, offset[0]:offset[0] + int(width / scale) + 2]
    crop = ndimage.zoom(crop, (height / crop.shape[0], width / crop.shape[1], 1), order=1)[:height, :width]

    lighting = ndimage.gaussian_filter(crop, (18, 18, 0)) + 1e-3
    return np.clip(crop / lighting, 0.4, 1.8)


def soft_shadow(alpha, offset, blur, strength):
    """The shadow an object with coverage `alpha` casts on what lies behind it."""
    shifted = ndimage.shift(alpha, offset, order=1)

    return np.clip(ndimage.gaussian_filter(shifted, blur) * strength, 0, 1)
