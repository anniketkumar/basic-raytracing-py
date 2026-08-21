"""Vectorized sphere intersection kernel (Week 2).

Solves ray-sphere quadratic equations across ALL (H, W) pixels simultaneously
using NumPy broadcasting, replacing the per-pixel loop in Sphere.intersect().

Broadcasting overview
---------------------
A single sphere intersection requires solving  a*t² + b*t + c = 0  for each ray.
Instead of iterating over H×W pixels, we express the coefficients as (H, W)-shaped
arrays and let NumPy evaluate them in one pass:

    ray_dirs:       (H, W, 3)   — one direction vector per pixel
    sphere_center:  (3,)        — a single point

    oc = ray_origins - center   — broadcasts (H, W, 3) - (3,) → (H, W, 3)
    a  = Σ(ray_dirs², axis=-1)  — contracts xyz → (H, W) dot products
    b  = 2·Σ(oc·dirs, axis=-1)  — same contraction
    c  = Σ(oc², axis=-1) - r²   — (H, W) scalars

Discriminant, roots, and masking are all (H, W) element-wise operations.
np.where acts as a branchless conditional: misses (disc < 0) are set to np.inf
without any Python-level branching.

After processing every sphere in a small Python loop, the per-sphere t arrays
are stacked to (num_spheres, H, W) and reduced with np.min / np.argmin along
axis=0 to find the closest hit per pixel.

Dependencies: NumPy only (no new packages).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sphere_to_dict(sphere):
    """Convert a core.sphere.Sphere object to the dict format used by intersect_spheres.

    Args:
        sphere: A Sphere instance with .position (Vec3), .radius (float),
                .color (tuple), and Blinn-Phong material properties.

    Returns:
        dict with keys:
            'center':    np.ndarray of shape (3,)
            'radius':    float
            'color':     tuple (R, G, B) in 0–255
            'ambient':   float — ambient reflection coefficient
            'diffuse':   float — diffuse reflection coefficient
            'specular':  float — specular reflection coefficient
            'shininess': float — specular exponent (higher = tighter highlight)
    """
    return {
        "center": np.array([sphere.position.x,
                            sphere.position.y,
                            sphere.position.z], dtype=np.float64),
        "radius": float(sphere.radius),
        "color": sphere.color,
        "ambient": float(getattr(sphere, 'ambient', 0.1)),
        "diffuse": float(getattr(sphere, 'diffuse', 0.7)),
        "specular": float(getattr(sphere, 'specular', 0.5)),
        "shininess": float(getattr(sphere, 'shininess', 32)),
    }


# ---------------------------------------------------------------------------
# Single-sphere kernel (internal)
# ---------------------------------------------------------------------------

def _intersect_single_sphere(ray_origins, ray_dirs, center, radius):
    """Solve the ray-sphere quadratic for ONE sphere across all (H, W) rays.

    The standard ray-sphere equation for a ray  P(t) = O + t·D  and a sphere
    with center C and radius r is:

        |P(t) - C|² = r²
        ⟹  a·t² + b·t + c = 0

    where:
        a = D·D        (per-ray dot product)
        b = 2·(O−C)·D  (per-ray dot product)
        c = (O−C)·(O−C) − r²

    Args:
        ray_origins: np.ndarray, shape (H, W, 3) — ray origin positions.
        ray_dirs:    np.ndarray, shape (H, W, 3) — unit ray directions.
        center:      np.ndarray, shape (3,)      — sphere centre.
        radius:      float                       — sphere radius.

    Returns:
        t: np.ndarray, shape (H, W) — nearest positive t for each ray.
           Pixels that miss the sphere (disc < 0) or have both roots ≤ 0
           (sphere behind camera) are set to np.inf.

    Shape trace:
        oc          (H, W, 3)   ray_origins - center  [broadcast (3,)]
        a           (H, W)      dot(ray_dirs, ray_dirs)
        b           (H, W)      2 * dot(oc, ray_dirs)
        c           (H, W)      dot(oc, oc) - radius²
        disc        (H, W)      b² - 4ac
        sqrt_disc   (H, W)      sqrt(max(disc, 0))  [safe for negative disc]
        t1, t2      (H, W)      the two quadratic roots
        t           (H, W)      nearest positive root, or np.inf
    """
    # --- Quadratic coefficients -------------------------------------------
    oc = ray_origins - center                           # (H, W, 3)
    a = np.sum(ray_dirs * ray_dirs, axis=-1)            # (H, W)
    b = 2.0 * np.sum(oc * ray_dirs, axis=-1)            # (H, W)
    c = np.sum(oc * oc, axis=-1) - radius * radius      # (H, W)

    # --- Discriminant & safe sqrt -----------------------------------------
    disc = b * b - 4.0 * a * c                          # (H, W)

    # np.sqrt of a negative value produces NaN; clamp to 0 so we can compute
    # roots everywhere and mask misses afterwards with np.where.
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))           # (H, W)

    # --- Both roots -------------------------------------------------------
    inv_2a = 1.0 / (2.0 * a)                            # (H, W)
    t1 = (-b - sqrt_disc) * inv_2a                       # (H, W)  — smaller root
    t2 = (-b + sqrt_disc) * inv_2a                       # (H, W)  — larger root

    # --- Choose the nearest POSITIVE root ---------------------------------
    # Priority: t1 if t1 > 0, else t2 if t2 > 0, else np.inf (miss / behind).
    INF = np.inf
    t = np.where(t1 > 0, t1,
         np.where(t2 > 0, t2, INF))                     # (H, W)

    # --- Mask pixels where the discriminant is negative (true misses) -----
    t = np.where(disc >= 0, t, INF)                      # (H, W)

    return t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def intersect_spheres(ray_origins, ray_dirs, spheres):
    """Test all rays against all spheres; return the closest hit per pixel.

    This is the main entry point for the vectorized intersection kernel.
    It loops over the (typically small) list of spheres and, for each one,
    evaluates the ray-sphere quadratic across every pixel simultaneously.
    The per-sphere t-value arrays are then stacked and reduced to find the
    closest hit.

    Args:
        ray_origins: np.ndarray, shape (H, W, 3).
                     For a pinhole camera every element is the same position,
                     but the signature accepts a full array for generality.
        ray_dirs:    np.ndarray, shape (H, W, 3) — unit direction vectors
                     (output of Camera.get_rays_vectorized).
        spheres:     list[dict] — each dict has:
                        'center': np.ndarray (3,)
                        'radius': float
                        'color':  tuple (R, G, B)

    Returns:
        t_min:          np.ndarray (H, W)    — closest-hit t value per pixel.
                        np.inf where no sphere was hit.
        hit_sphere_idx: np.ndarray (H, W), dtype int — index into `spheres`
                        for the closest sphere, or -1 for misses.
        normals:        np.ndarray (H, W, 3) — outward surface normal at the
                        hit point. Zero vector for miss pixels.
        hit_points:     np.ndarray (H, W, 3) — world-space hit positions.
                        Zero vector for miss pixels.

    Shape trace (multi-sphere reduction):
        all_t           (num_spheres, H, W)  stacked per-sphere t arrays
        t_min           (H, W)               np.min(all_t, axis=0)
        hit_sphere_idx  (H, W)               np.argmin(all_t, axis=0)
        hit_points      (H, W, 3)            origin + t[..., None] * dirs
        normals         (H, W, 3)            (hit_point - center) / radius
    """
    H, W = ray_dirs.shape[0], ray_dirs.shape[1]
    num_spheres = len(spheres)

    # ------------------------------------------------------------------
    # 1. Solve the quadratic for every sphere → (num_spheres, H, W)
    # ------------------------------------------------------------------
    all_t = np.empty((num_spheres, H, W), dtype=np.float64)
    for i, s in enumerate(spheres):
        all_t[i] = _intersect_single_sphere(
            ray_origins, ray_dirs, s["center"], s["radius"]
        )

    # ------------------------------------------------------------------
    # 2. Multi-sphere closest-hit reduction
    # ------------------------------------------------------------------
    t_min = np.min(all_t, axis=0)              # (H, W) — closest t per pixel
    hit_sphere_idx = np.argmin(all_t, axis=0)  # (H, W) — winning sphere index

    # Mark miss pixels (no sphere hit) with index -1.
    miss_mask = (t_min == np.inf)               # (H, W) bool
    hit_sphere_idx = np.where(miss_mask, -1, hit_sphere_idx)  # (H, W) int

    # ------------------------------------------------------------------
    # 3. Hit-point computation
    #    origin + t * direction,  with t expanded to (H, W, 1) for broadcast
    # ------------------------------------------------------------------
    # Use a safe copy of t_min where inf is replaced by 0 so that the
    # multiplication doesn't produce inf * 0 = NaN warnings.  Miss pixels
    # are zeroed out explicitly in step 5.
    safe_t = np.where(miss_mask, 0.0, t_min)                      # (H, W)
    hit_points = ray_origins + safe_t[..., np.newaxis] * ray_dirs  # (H, W, 3)

    # ------------------------------------------------------------------
    # 4. Surface normals — gathered per winning sphere
    #    For each sphere index i, wherever hit_sphere_idx == i:
    #        normal = (hit_point - center_i) / radius_i
    # ------------------------------------------------------------------
    normals = np.zeros((H, W, 3), dtype=np.float64)
    for i, s in enumerate(spheres):
        mask = (hit_sphere_idx == i)             # (H, W) bool
        if not np.any(mask):
            continue
        # mask[..., np.newaxis] broadcasts (H, W, 1) against (H, W, 3)
        diff = hit_points - s["center"]          # (H, W, 3)
        sphere_normals = diff / s["radius"]      # (H, W, 3) — unit normal
        normals = np.where(mask[..., np.newaxis], sphere_normals, normals)

    # ------------------------------------------------------------------
    # 5. Zero out miss pixels
    # ------------------------------------------------------------------
    hit_points = np.where(miss_mask[..., np.newaxis], 0.0, hit_points)
    normals    = np.where(miss_mask[..., np.newaxis], 0.0, normals)

    return t_min, hit_sphere_idx, normals, hit_points
