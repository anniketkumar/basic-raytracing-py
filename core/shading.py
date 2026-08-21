"""Vectorized Blinn-Phong shading kernel (Week 4).

Upgrades the Week 3 Lambertian diffuse-only shading to the full Blinn-Phong
lighting model with per-sphere material properties:

    final = ambient + Σ_lights ( diffuse_term + specular_term )

    ambient_term  = ambient_coeff * sphere_color
    diffuse_term  = sphere_color * light_color * NdotL * diffuse_coeff * intensity
    specular_term = light_color * (NdotH ** shininess) * specular_coeff * intensity

The Blinn-Phong half-vector  H = normalize(L + V)  is cheaper than computing
the reflection vector and gives nearly identical results.

Broadcasting overview for the new terms
---------------------------------------
    view_dir:    normalize(ray_origins - hit_points)     → (H, W, 3)
    half_vec:    normalize(light_dir + view_dir)         → (H, W, 3)
    NdotH:       max(dot(normals, half_vec), 0)          → (H, W)
    specular:    NdotH ** shininess * specular_coeff     → (H, W)

Material coefficients are gathered per-pixel using the same
``mask = (hit_sphere_idx == i)`` pattern used for sphere colours in Week 3.

Dependencies: NumPy only (no new packages).
"""

import numpy as np

from core.intersection import _intersect_single_sphere


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def light_to_dict(light):
    """Convert a core.light.Light object to the dict format used by shading.

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

def shade_blinn_phong(hit_points, normals, hit_sphere_idx, t_min,
                      spheres, lights, ray_origins=None):
    """Compute Blinn-Phong shading (ambient + diffuse + specular) for all pixels.

    This function extends the Week 3 Lambertian diffuse shading with:
        - Per-sphere ambient illumination (prevents pure-black shadows)
        - Specular highlights via the Blinn-Phong half-vector method
        - Per-sphere material properties (ambient, diffuse, specular, shininess)

    If ``ray_origins`` is None, specular highlights are skipped and the result
    is equivalent to the original shade_diffuse (used for backward compat).

    Args:
        hit_points:     np.ndarray (H, W, 3) — world-space hit positions.
        normals:        np.ndarray (H, W, 3) — unit outward surface normals.
        hit_sphere_idx: np.ndarray (H, W)    — index into ``spheres``
                        (-1 = miss).
        t_min:          np.ndarray (H, W)    — primary-ray hit distance
                        (np.inf for misses).
        spheres:        list[dict] — sphere dicts with keys ``center`` (3,),
                        ``radius`` (float), ``color`` (tuple), ``ambient``,
                        ``diffuse``, ``specular``, ``shininess``.
        lights:         list[dict] — light dicts with keys ``position`` (3,),
                        ``intensity`` (float), ``color`` (tuple).
        ray_origins:    np.ndarray (H, W, 3) or None — ray origin positions.
                        Required for specular; if None, specular is disabled.

    Returns:
        image: np.ndarray (H, W, 3), dtype uint8 — final RGB pixel values,
               clamped to [0, 255].

    Shape trace:
        sphere_colors   (H, W, 3)   per-pixel object colour, float [0, 1]
        ambient_coeff   (H, W)      per-pixel ambient material coefficient
        diffuse_coeff   (H, W)      per-pixel diffuse material coefficient
        specular_coeff  (H, W)      per-pixel specular material coefficient
        shininess       (H, W)      per-pixel specular exponent
        light_vec       (H, W, 3)   unnormalised hit→light vector
        light_dir       (H, W, 3)   normalised light direction
        NdotL           (H, W)      Lambert cosine term, clamped >= 0
        view_dir        (H, W, 3)   normalised camera→hit direction (reversed)
        half_vec        (H, W, 3)   normalised bisector of L and V
        NdotH           (H, W)      specular cosine term, clamped >= 0
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

    # Early exit: no geometry → black image.
    # Note: unlike Week 3, we DON'T early-exit on no lights because ambient
    # illumination doesn't require any light sources.
    if num_spheres == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    # 1. Build per-pixel sphere colour and material properties
    #
    #    For each sphere index i, wherever hit_sphere_idx == i, assign
    #    that sphere's colour and material coefficients.  Miss pixels
    #    remain at zero (black / no contribution).
    # ------------------------------------------------------------------
    sphere_colors  = np.zeros((H, W, 3), dtype=np.float64)
    ambient_coeff  = np.zeros((H, W),    dtype=np.float64)
    diffuse_coeff  = np.zeros((H, W),    dtype=np.float64)
    specular_coeff = np.zeros((H, W),    dtype=np.float64)
    shininess      = np.ones((H, W),     dtype=np.float64)  # default 1 to avoid 0**0

    for i, s in enumerate(spheres):
        mask = (hit_sphere_idx == i)                            # (H, W) bool
        if not np.any(mask):
            continue
        color_01 = np.array(s["color"], dtype=np.float64) / 255.0  # (3,)
        sphere_colors[mask]  = color_01
        ambient_coeff[mask]  = s.get("ambient",  0.1)
        diffuse_coeff[mask]  = s.get("diffuse",  0.7)
        specular_coeff[mask] = s.get("specular", 0.5)
        shininess[mask]      = s.get("shininess", 32)

    # Miss mask — pixels that didn't hit any sphere
    miss_mask = (hit_sphere_idx == -1)                          # (H, W)

    # ------------------------------------------------------------------
    # 2. Ambient term  (computed once, outside the light loop)
    #    ambient_color = ambient_coeff * sphere_color
    # ------------------------------------------------------------------
    ambient_color = sphere_colors * ambient_coeff[..., np.newaxis]  # (H, W, 3)

    # ------------------------------------------------------------------
    # 3. Compute view direction for specular (if ray_origins provided)
    # ------------------------------------------------------------------
    compute_specular = (ray_origins is not None)
    if compute_specular:
        view_vec = ray_origins - hit_points                      # (H, W, 3)
        view_dist = np.linalg.norm(view_vec, axis=-1,
                                   keepdims=True)                # (H, W, 1)
        view_dir = view_vec / np.maximum(view_dist, 1e-10)      # (H, W, 3)

    # Pre-compute sphere-index comparison array for self-exclusion.
    # sphere_indices[i, :, :] == i, compared against hit_sphere_idx via
    # broadcasting:  (N, 1, 1) == (1, H, W)  ->  (N, H, W)
    sphere_indices = np.arange(num_spheres).reshape(num_spheres, 1, 1)

    # ------------------------------------------------------------------
    # 4. Per-light diffuse + specular accumulation
    # ------------------------------------------------------------------
    accumulated = np.zeros((H, W, 3), dtype=np.float64)

    if len(lights) == 0:
        # No lights → only ambient contributes
        image = np.clip(ambient_color * 255.0, 0.0, 255.0).astype(np.uint8)
        return image

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

        # Zero out terms where light is blocked or pixel is a miss
        visible = ~(shadow_blocked | miss_mask)                 # (H, W)
        NdotL_masked = np.where(visible, NdotL, 0.0)           # (H, W)

        # --- Diffuse contribution -----------------------------------------
        # sphere_color * light_color * NdotL * diffuse_coeff * intensity
        diffuse_contrib = (
            sphere_colors
            * light_color_01
            * (NdotL_masked * diffuse_coeff)[..., np.newaxis]
            * light_intensity
        )                                                       # (H, W, 3)

        # --- Specular contribution ----------------------------------------
        if compute_specular:
            # Half vector: H = normalize(L + V)
            half_vec = light_dir + view_dir                     # (H, W, 3)
            half_dist = np.linalg.norm(half_vec, axis=-1,
                                       keepdims=True)           # (H, W, 1)
            half_vec = half_vec / np.maximum(half_dist, 1e-10)  # (H, W, 3)

            # NdotH = max(dot(normals, half_vec), 0)
            NdotH = np.sum(normals * half_vec, axis=-1)         # (H, W)
            NdotH = np.maximum(NdotH, 0.0)                     # (H, W)

            # Specular intensity: (NdotH ** shininess) * specular_coeff
            spec_intensity = (NdotH ** shininess) * specular_coeff  # (H, W)
            spec_intensity = np.where(visible, spec_intensity, 0.0)

            # Specular uses LIGHT colour, not sphere colour (standard Blinn-Phong)
            specular_contrib = (
                light_color_01
                * spec_intensity[..., np.newaxis]
                * light_intensity
            )                                                   # (H, W, 3)
        else:
            specular_contrib = 0.0

        accumulated += diffuse_contrib + specular_contrib

    # ------------------------------------------------------------------
    # 5. Final pixel colour: ambient + sum of light contributions
    #
    #    Matches the scalar:  min(255, int(c * 255))
    #    np.clip prevents uint8 overflow; .astype truncates like int().
    # ------------------------------------------------------------------
    final = ambient_color + accumulated
    image = np.clip(final * 255.0, 0.0, 255.0).astype(np.uint8)

    return image


def shade_diffuse(hit_points, normals, hit_sphere_idx, t_min, spheres, lights):
    """Backward-compatible alias for shade_blinn_phong without specular.

    Calls shade_blinn_phong with ray_origins=None, which disables the specular
    term.  The ambient + diffuse result matches the Week 3 Lambertian shading
    when spheres use the default material properties (ambient=0.1, diffuse=0.7).

    .. note::
        To get exact Week 3 parity (diffuse-only, no ambient), the sphere dicts
        must have ambient=0.  With the default ambient=0.1, shadowed regions
        will gain a faint ambient contribution.
    """
    return shade_blinn_phong(
        hit_points, normals, hit_sphere_idx, t_min,
        spheres, lights, ray_origins=None
    )
