"""Validation suite for the vectorized sphere intersection kernel (Week 2).

Tests the functions in core.intersection against known geometric configurations
and against the original scalar Sphere.intersect() method for parity.

Usage:
    python -m tests.test_intersection
"""

import sys
import os
import math
import numpy as np

# Ensure the project root is on sys.path so `core.*` imports resolve.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.intersection import intersect_spheres, sphere_to_dict, _intersect_single_sphere
from core.sphere import Sphere
from core.vec3 import Vec3
from core.ray import Ray
from core.camera import Camera


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_single_ray(origin_np, dir_np):
    """Build (1, 1, 3) origin and direction arrays from plain (3,) vectors."""
    o = origin_np.reshape(1, 1, 3).astype(np.float64)
    d = dir_np / np.linalg.norm(dir_np)  # normalise
    d = d.reshape(1, 1, 3).astype(np.float64)
    return o, d


# ---------------------------------------------------------------------------
# Test 1: Direct hit — ray aimed straight at a sphere's centre
# ---------------------------------------------------------------------------

def test_direct_hit():
    """A ray originating at the origin and pointing along +Z should hit a
    sphere centred on the +Z axis.

    Expected: t > 0, finite; hit_sphere_idx == 0; normal points back at camera.
    """
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    ray_o, ray_d = _make_single_ray(origin, direction)

    sphere = {"center": np.array([0.0, 0.0, 5.0]), "radius": 1.0, "color": (255, 0, 0)}

    t_min, idx, normals, hit_pts = intersect_spheres(ray_o, ray_d, [sphere])

    t_val = t_min[0, 0]
    assert np.isfinite(t_val), f"Expected finite t, got {t_val}"
    assert t_val > 0, f"Expected t > 0, got {t_val}"
    assert idx[0, 0] == 0, f"Expected sphere index 0, got {idx[0, 0]}"

    # The hit should be at t = 5 - 1 = 4 (front surface of unit sphere at z=5)
    assert np.isclose(t_val, 4.0), f"Expected t ≈ 4.0, got {t_val}"

    # Normal at the front surface should point back toward camera (-Z)
    expected_normal = np.array([0.0, 0.0, -1.0])
    assert np.allclose(normals[0, 0], expected_normal, atol=1e-12), \
        f"Expected normal {expected_normal}, got {normals[0, 0]}"

    return True


# ---------------------------------------------------------------------------
# Test 2: Complete miss — ray aimed away from all spheres
# ---------------------------------------------------------------------------

def test_miss():
    """A ray pointing along +X should miss a sphere that sits on the +Z axis."""
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])  # pointing right, sphere is ahead
    ray_o, ray_d = _make_single_ray(origin, direction)

    sphere = {"center": np.array([0.0, 0.0, 10.0]), "radius": 1.0, "color": (0, 255, 0)}

    t_min, idx, normals, hit_pts = intersect_spheres(ray_o, ray_d, [sphere])

    assert t_min[0, 0] == np.inf, f"Expected t == inf (miss), got {t_min[0, 0]}"
    assert idx[0, 0] == -1, f"Expected sphere index -1 (miss), got {idx[0, 0]}"
    assert np.allclose(normals[0, 0], 0.0), \
        f"Expected zero normal for miss, got {normals[0, 0]}"

    return True


# ---------------------------------------------------------------------------
# Test 3: Closer sphere wins — two overlapping spheres
# ---------------------------------------------------------------------------

def test_closer_sphere_wins():
    """Two spheres along the +Z axis; the nearer one should win."""
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    ray_o, ray_d = _make_single_ray(origin, direction)

    near_sphere = {"center": np.array([0.0, 0.0, 5.0]),  "radius": 1.0, "color": (255, 0, 0)}
    far_sphere  = {"center": np.array([0.0, 0.0, 10.0]), "radius": 1.0, "color": (0, 0, 255)}

    # Pass the far sphere FIRST to verify argmin picks the near one regardless of order.
    t_min, idx, normals, _ = intersect_spheres(ray_o, ray_d, [far_sphere, near_sphere])

    # near_sphere is at index 1 in the list
    assert idx[0, 0] == 1, f"Expected closer sphere index 1, got {idx[0, 0]}"
    assert np.isclose(t_min[0, 0], 4.0), \
        f"Expected t ≈ 4.0 (front of near sphere), got {t_min[0, 0]}"

    return True


# ---------------------------------------------------------------------------
# Test 4: Tangent ray — discriminant ≈ 0
# ---------------------------------------------------------------------------

def test_tangent_ray():
    """A ray that just grazes the sphere (disc ≈ 0) should still produce a
    valid, finite t value — a single contact point.
    """
    # Sphere at origin, radius 1. A ray starting at (-10, 1, 0) heading +X
    # passes through y=1, which is exactly on the sphere surface.
    origin = np.array([-10.0, 1.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    ray_o, ray_d = _make_single_ray(origin, direction)

    sphere = {"center": np.array([0.0, 0.0, 0.0]), "radius": 1.0, "color": (255, 255, 0)}

    t_min, idx, normals, hit_pts = intersect_spheres(ray_o, ray_d, [sphere])

    t_val = t_min[0, 0]
    assert np.isfinite(t_val), f"Tangent ray should hit (finite t), got {t_val}"
    assert t_val > 0, f"Tangent t should be positive, got {t_val}"
    assert idx[0, 0] == 0, f"Expected sphere index 0, got {idx[0, 0]}"

    # The tangent point should be at (0, 1, 0) — distance 10 from origin
    assert np.isclose(t_val, 10.0, atol=1e-9), \
        f"Expected t ≈ 10.0 for tangent, got {t_val}"
    assert np.allclose(hit_pts[0, 0], [0.0, 1.0, 0.0], atol=1e-9), \
        f"Expected hit at (0,1,0), got {hit_pts[0, 0]}"

    return True


# ---------------------------------------------------------------------------
# Test 5: Parity with scalar Sphere.intersect()
# ---------------------------------------------------------------------------

def test_parity_with_scalar():
    """Compare vectorized t values against the old per-pixel Sphere.intersect()
    on a small scene (4×4 image, 2 spheres).

    The tolerance accounts for floating-point ordering differences between
    scalar and vectorized paths.
    """
    W, H = 4, 4
    cam = Camera(
        position=Vec3(0, 0, -20),
        look_at=Vec3(0, 0, 0),
        fov_deg=60,
        aspect_ratio=W / H,
    )

    sphere_objs = [
        Sphere(Vec3(0, 0, 5), 2, (255, 0, 0)),
        Sphere(Vec3(3, 0, 8), 1.5, (0, 255, 0)),
    ]

    # --- Vectorized path ---
    ray_dirs = cam.get_rays_vectorized(W, H)                     # (H, W, 3)
    cam_pos_np = np.array([cam.position.x, cam.position.y, cam.position.z])
    ray_origins = np.broadcast_to(cam_pos_np, ray_dirs.shape).copy()  # (H, W, 3)

    sphere_dicts = [sphere_to_dict(s) for s in sphere_objs]
    t_vec, idx_vec, _, _ = intersect_spheres(ray_origins, ray_dirs, sphere_dicts)

    # --- Scalar (loop) path ---
    t_scalar = np.full((H, W), np.inf, dtype=np.float64)
    idx_scalar = np.full((H, W), -1, dtype=np.int64)

    for y in range(H):
        for x in range(W):
            d = ray_dirs[y, x]
            ray = Ray(cam.position, Vec3(d[0], d[1], d[2]))
            best_t = np.inf
            best_i = -1
            for si, s in enumerate(sphere_objs):
                hit = s.intersect(ray)
                if hit is not None:
                    t_hit = hit[0]
                    if 0 < t_hit < best_t:
                        best_t = t_hit
                        best_i = si
            t_scalar[y, x] = best_t
            idx_scalar[y, x] = best_i

    # --- Compare ---
    # For miss pixels both should be inf; for hits they should be numerically close.
    t_close = np.allclose(t_vec, t_scalar, atol=1e-9, equal_nan=False)
    idx_match = np.array_equal(idx_vec, idx_scalar)

    assert t_close, (
        f"t-value mismatch!\n"
        f"  Max diff: {np.max(np.abs(np.where(np.isinf(t_vec) & np.isinf(t_scalar), 0, t_vec - t_scalar))):.2e}\n"
        f"  Vectorized:\n{t_vec}\n"
        f"  Scalar:\n{t_scalar}"
    )
    assert idx_match, (
        f"Sphere-index mismatch!\n"
        f"  Vectorized:\n{idx_vec}\n"
        f"  Scalar:\n{idx_scalar}"
    )

    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("Direct hit",              test_direct_hit),
    ("Complete miss",           test_miss),
    ("Closer sphere wins",     test_closer_sphere_wins),
    ("Tangent ray (disc ~= 0)", test_tangent_ray),
    ("Parity with scalar",     test_parity_with_scalar),
]


def run_tests():
    print("=" * 70)
    print("  Vectorized Sphere Intersection — Validation Suite (Week 2)")
    print("=" * 70)

    passed = 0
    failed = 0

    for name, fn in TESTS:
        try:
            fn()
            print(f"  [PASS]  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL]  {name}")
            print(f"          {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {name}")
            print(f"          {type(e).__name__}: {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"  {passed} passed, {failed} failed out of {passed + failed}")
    if failed == 0:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
