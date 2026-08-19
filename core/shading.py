"""Vectorized Lambertian diffuse shading kernel (Week 3).

Given the (H, W, 3) hit points, normals, and per-pixel sphere indices from
the Week 2 intersection kernel, this module computes per-pixel Lambertian
diffuse lighting from all scene lights in one batched pass — including
vectorized shadow rays.

Broadcasting overview for shading
---------------------------------
Each light contributes illumination to every hit pixel.  The key operations:

    light_pos:       (3,)         — single light position
    hit_points:      (H, W, 3)   — per-pixel hit locations

    light_vec = light_pos - hit_points    — broadcasts (3,) against (H, W, 3)
    NdotL = sum(normals * light_dir, -1)  — contracts xyz -> (H, W) scalars
    contribution = color * NdotL[..., None] — (H, W, 1) broadcasts to (H, W, 3)

Shadow rays reuse the Week 2 single-sphere intersection kernel:  shadow
origins and directions are (H, W, 3) arrays fed directly to
_intersect_single_sphere.  Self-sphere exclusion is handled per-pixel by
building a (num_spheres, H, W) boolean mask where ``sphere_index == i`` and
setting those shadow-t entries to inf before taking the minimum.

Dependencies: NumPy only (no new packages).
"""

import numpy as np

from core.intersection import _intersect_single_sphere


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def light_to_dict(light):
    """Convert a core.light.Light object to the dict format used by shade_diffuse.

    Args:
        light: A Light instance with .position (Vec3), .intensity (float),
               and .color (tuple).

    Returns:
        dict with keys:
            'position':  np.ndarray of shape (3,)
            'intensity': float
            'color':     tuple (R, G, B) in 0-255
    """
    return {
        "position": np.array([light.position.x,
                              light.position.y,
                              light.position.z], dtype=np.float64),
        "intensity": float(light.intensity),
        "color": light.color,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def shade_diffuse(hit_points, normals, hit_sphere_idx, t_min, spheres, lights):
    """Compute Lambertian diffuse shading for all pixels simultaneously.

    This function replaces the per-pixel colour accumulation loop in
    DemoViewer.trace().  For each light in the scene it:
        1. Computes light-direction vectors for all (H, W) hit points
        2. Evaluates the Lambert cosine term  max(N . L, 0)
        3. Fires vectorized shadow rays (reusing the Week 2 kernel)
        4. Accumulates  sphere_color * light_color * NdotL * intensity

    The colour math exactly mirrors the scalar trace() implementation::

        diff = max(normal.dot(ldir), 0) * light.intensity
        color[i] += (s.color[i] / 255) * (light.color[i] / 255) * diff
        final = min(255, int(color_float * 255))

    Args:
        hit_points:     np.ndarray (H, W, 3) — world-space hit positions.
        normals:        np.ndarray (H, W, 3) — unit outward surface normals.
        hit_sphere_idx: np.ndarray (H, W)    — index into ``spheres``
                        (-1 = miss).
        t_min:          np.ndarray (H, W)    — primary-ray hit distance
                        (np.inf for misses).
        spheres:        list[dict] — sphere dicts with keys ``center`` (3,),
                        ``radius`` (float), ``color`` (tuple).
        lights:         list[dict] — light dicts with keys ``position`` (3,),
                        ``intensity`` (float), ``color`` (tuple).

    Returns:
        image: np.ndarray (H, W, 3), dtype uint8 — final RGB pixel values,
               clamped to [0, 255].

    Shape trace:
        sphere_colors   (H, W, 3)   per-pixel object colour, float [0, 1]
        light_vec       (H, W, 3)   unnormalised hit->light vector
        light_dir       (H, W, 3)   normalised light direction
        NdotL           (H, W)      Lambert cosine term, clamped >= 0
        shadow_origins  (H, W, 3)   offset hit points for shadow rays
        all_shadow_t    (N, H, W)   per-sphere shadow-ray t values
        self_mask       (N, H, W)   True where sphere i is the hit sphere
        shadow_t_min    (H, W)      closest non-self shadow hit
        shadow_blocked  (H, W)      bool — True where light is occluded
        contribution    (H, W, 3)   per-light colour contribution
        accumulated     (H, W, 3)   sum of all light contributions
        image           (H, W, 3)   uint8 final image
    """
    H, W = hit_sphere_idx.shape
    num_spheres = len(spheres)

    # Early exit: no geometry or no lights -> black image
    if num_spheres == 0 or len(lights) == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    # 1. Build per-pixel sphere colour  (H, W, 3), float in [0, 1]
    #
    #    For each sphere index i, wherever hit_sphere_idx == i, assign
    #    that sphere's colour normalised to [0, 1].  Miss pixels remain
    #    at zero (black).
    # ------------------------------------------------------------------
    sphere_colors = np.zeros((H, W, 3), dtype=np.float64)
    for i, s in enumerate(spheres):
        mask = (hit_sphere_idx == i)                            # (H, W) bool
        if not np.any(mask):
            continue
        color_01 = np.array(s["color"], dtype=np.float64) / 255.0  # (3,)
        sphere_colors[mask] = color_01

    # Miss mask — pixels that didn't hit any sphere
    miss_mask = (hit_sphere_idx == -1)                          # (H, W)

    # Pre-compute sphere-index comparison array for self-exclusion.
    # sphere_indices[i, :, :] == i, compared against hit_sphere_idx via
    # broadcasting:  (N, 1, 1) == (1, H, W)  ->  (N, H, W)
    sphere_indices = np.arange(num_spheres).reshape(num_spheres, 1, 1)

    # ------------------------------------------------------------------
    # 2. Per-light diffuse accumulation
    # ------------------------------------------------------------------
    accumulated = np.zeros((H, W, 3), dtype=np.float64)

    for light in lights:
        light_pos       = light["position"]                     # (3,)
        light_intensity = light["intensity"]                    # scalar
        light_color_01  = np.array(light["color"],
                                   dtype=np.float64) / 255.0   # (3,)

        # --- Light direction ----------------------------------------------
        # Broadcast: (3,) - (H, W, 3)  ->  (H, W, 3)
        light_vec  = light_pos - hit_points                     # (H, W, 3)
        light_dist = np.linalg.norm(light_vec, axis=-1,
                                    keepdims=True)              # (H, W, 1)
        # Safe normalisation (avoid /0 for miss pixels where hit_points=0)
        light_dir  = light_vec / np.maximum(light_dist, 1e-10) # (H, W, 3)

        # --- Lambert cosine term ------------------------------------------
        # Contracts xyz axis: sum of element-wise product -> (H, W) scalars
        NdotL = np.sum(normals * light_dir, axis=-1)            # (H, W)
        NdotL = np.maximum(NdotL, 0.0)                         # (H, W)

        # --- Vectorized shadow rays ---------------------------------------
        # Offset origin along surface normal to prevent self-intersection
        # (same epsilon = 0.01 as the scalar trace())
        SHADOW_EPSILON = 0.01
        shadow_origins = hit_points + normals * SHADOW_EPSILON  # (H, W, 3)

        # Solve shadow intersection for EACH sphere individually so we
        # can exclude the self-sphere on a per-pixel basis.
        all_shadow_t = np.full((num_spheres, H, W), np.inf,
                               dtype=np.float64)
        for i, s in enumerate(spheres):
            all_shadow_t[i] = _intersect_single_sphere(
                shadow_origins, light_dir, s["center"], s["radius"]
            )

        # Self-exclusion: where the primary hit sphere is sphere i, set
        # that sphere's shadow t to inf so it can't block itself.
        # Broadcasting: (N, 1, 1) == (1, H, W)  ->  (N, H, W)
        self_mask = (sphere_indices == hit_sphere_idx[np.newaxis, :, :])
        all_shadow_t = np.where(self_mask, np.inf, all_shadow_t)

        # Closest shadow hit (excluding self)
        shadow_t_min = np.min(all_shadow_t, axis=0)             # (H, W)
        shadow_blocked = (shadow_t_min < np.inf)                # (H, W)

        # Zero out NdotL where light is blocked or pixel is a miss
        NdotL = np.where(shadow_blocked | miss_mask, 0.0, NdotL)

        # --- Colour contribution ------------------------------------------
        # Mirrors the scalar formula:
        #   color[c] += (s.color[c]/255) * (l.color[c]/255) * NdotL * intensity
        #
        # Broadcasting:
        #   (H,W,3) * (3,) -> (H,W,3)   [sphere_colors * light_color_01]
        #   (H,W,3) * (H,W,1) -> (H,W,3) [... * NdotL[..., None]]
        #   (H,W,3) * scalar -> (H,W,3)  [... * light_intensity]
        contribution = (
            sphere_colors
            * light_color_01
            * NdotL[..., np.newaxis]
            * light_intensity
        )                                                       # (H, W, 3)
        accumulated += contribution

    # ------------------------------------------------------------------
    # 3. Final conversion: float [0, inf) -> uint8 [0, 255]
    #
    #    Matches the scalar:  min(255, int(c * 255))
    #    np.clip prevents uint8 overflow; .astype truncates like int().
    # ------------------------------------------------------------------
    image = np.clip(accumulated * 255.0, 0.0, 255.0).astype(np.uint8)

    return image
