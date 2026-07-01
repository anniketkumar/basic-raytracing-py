"""
Validation script: old loop-based ray generation vs. new vectorized version.

Runs both Camera.get_ray() (per-pixel) and Camera.get_rays_vectorized() (batched)
on identical camera setups, then asserts the resulting direction arrays are equal
within floating-point tolerance.

Usage:
    python -m tests.test_vectorized_rays
"""

import sys
import os
import time
import numpy as np

# Ensure the project root is on sys.path so `core.*` imports resolve.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.camera import Camera
from core.vec3 import Vec3


def build_loop_rays(camera, W, H):
    """Generate ray directions using the original per-pixel loop.

    Returns:
        np.ndarray: shape (H, W, 3) — the same layout as get_rays_vectorized().
    """
    result = np.empty((H, W, 3), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            d = camera.get_ray(x, y, W, H)
            result[y, x] = [d.x, d.y, d.z]
    return result


# ---------------------------------------------------------------------------
# Test configurations — exercise different positions, orientations, FOVs,
# aspect ratios, and image sizes.
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "name": "Default scene camera (300x300)",
        "camera": Camera(
            position=Vec3(0, 0, -200),
            look_at=Vec3(0, 0, 50),
            fov_deg=60,
            aspect_ratio=300 / 300,
        ),
        "width": 300,
        "height": 300,
    },
    {
        "name": "Wide-angle 16:9 (160x90)",
        "camera": Camera(
            position=Vec3(10, -5, -100),
            look_at=Vec3(0, 0, 0),
            fov_deg=90,
            aspect_ratio=16 / 9,
        ),
        "width": 160,
        "height": 90,
    },
    {
        "name": "Narrow FOV portrait (90x160)",
        "camera": Camera(
            position=Vec3(-30, 20, -50),
            look_at=Vec3(10, -10, 100),
            fov_deg=30,
            aspect_ratio=90 / 160,
        ),
        "width": 90,
        "height": 160,
    },
    {
        "name": "Edge case: 1x1 image",
        "camera": Camera(
            position=Vec3(0, 0, 0),
            look_at=Vec3(0, 0, 1),
            fov_deg=60,
            aspect_ratio=1.0,
        ),
        "width": 1,
        "height": 1,
    },
    {
        "name": "Large image (800x600)",
        "camera": Camera(
            position=Vec3(0, 0, -200),
            look_at=Vec3(0, 0, 50),
            fov_deg=60,
            aspect_ratio=800 / 600,
        ),
        "width": 800,
        "height": 600,
    },
]


def run_tests():
    all_passed = True
    print("=" * 70)
    print("  Vectorized Ray Generation — Validation Suite")
    print("=" * 70)

    for tc in TEST_CASES:
        name = tc["name"]
        cam = tc["camera"]
        W, H = tc["width"], tc["height"]
        total_pixels = W * H

        print(f"\n--- {name} ({W}x{H} = {total_pixels:,} pixels) ---")

        # Old: per-pixel loop
        t0 = time.perf_counter()
        loop_dirs = build_loop_rays(cam, W, H)
        t_loop = time.perf_counter() - t0

        # New: vectorized
        t0 = time.perf_counter()
        vec_dirs = cam.get_rays_vectorized(W, H)
        t_vec = time.perf_counter() - t0

        # Compare
        max_abs_diff = np.max(np.abs(loop_dirs - vec_dirs))
        close = np.allclose(loop_dirs, vec_dirs, atol=1e-12)

        # Report
        print(f"  Loop time:        {t_loop:.4f}s")
        print(f"  Vectorized time:  {t_vec:.6f}s")
        if t_loop > 0:
            print(f"  Speedup:          {t_loop / t_vec:.1f}x")
        print(f"  Max abs diff:     {max_abs_diff:.2e}")
        print(f"  np.allclose:      {'PASS' if close else 'FAIL'}")

        if not close:
            all_passed = False
            # Show first mismatch for debugging
            diff = np.abs(loop_dirs - vec_dirs)
            idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"  First mismatch at index {idx}:")
            print(f"    loop:       {loop_dirs[idx[0], idx[1]]}")
            print(f"    vectorized: {vec_dirs[idx[0], idx[1]]}")

    print("\n" + "=" * 70)
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_tests())
