"""Validation suite for the vectorized diffuse shading kernel (Week 3).

Tests the shade_diffuse function against known geometric configurations and
against the original scalar trace() logic for parity.

Usage:
    python -m tests.test_shading
"""

import sys
import os
import numpy as np

# Ensure the project root is on sys.path so `core.*` imports resolve.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.intersection import intersect_spheres, sphere_to_dict, _intersect_single_sphere
from core.shading import shade_diffuse, light_to_dict
from core.sphere import Sphere
from core.light import Light
from core.vec3 import Vec3
from core.ray import Ray
from core.camera import Camera


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scalar_trace(ray_origin_v3, ray_dir_v3, sphere_objs, light_objs):
    """Replicate the scalar trace() logic from viewer.py for parity testing.

    This is a standalone copy of the per-pixel shading path so we can compare
    its output against the vectorized shade_diffuse without requiring pygame.

    Returns:
        tuple (R, G, B) in 0-255, matching DemoViewer.trace() output.
    """
    ray = Ray(ray_origin_v3, ray_dir_v3)

    # Find closest sphere hit
    closest = None
    min_t = float("inf")
    for s in sphere_objs:
        hit = s.intersect(ray)
        if hit and hit[0] < min_t:
            min_t = hit[0]
            closest = (s, hit)

    if not closest:
        return (0, 0, 0)

    s, (t, hit_point, normal) = closest
    color = [0.0, 0.0, 0.0]

    for light in light_objs:
        ldir = (light.position - hit_point).normalize()
        # Shadow ray
        shadow_ray = Ray(hit_point + normal * 0.01, ldir)
        blocked = False
        for o in sphere_objs:
            if o is not s and o.intersect(shadow_ray):
                blocked = True
                break
        if not blocked:
            diff = max(normal.dot(ldir), 0) * light.intensity
            for i in range(3):
                color[i] += (s.color[i] / 255) * (light.color[i] / 255) * diff

    return tuple(min(255, int(c * 255)) for c in color)


# ---------------------------------------------------------------------------
# Test 1: Single light, no shadows
# ---------------------------------------------------------------------------

def test_single_light_no_shadows():
    """One sphere, one light, no occlusion.

    Setup:
        - Ray from (0,0,-10) toward (0,0,1) hits sphere at (0,0,4)
        - Normal at front surface: (0,0,-1)
        - Light at (0,0,-10): light_dir from hit = (0,0,-1)
        - N.L = 1.0  ->  full illumination
        - Sphere color (255, 0, 0), light color (255, 255, 255), intensity 1.0
        - Expected: R = int(1.0 * 1.0 * 1.0 * 1.0 * 255) = 255, G=0, B=0
    """
    # Construct inputs manually for a single pixel
    ray_origins = np.array([[[0, 0, -10]]], dtype=np.float64)
    ray_dirs = np.array([[[0, 0, 1]]], dtype=np.float64)

    sphere = {"center": np.array([0, 0, 5.0]), "radius": 1.0, "color": (255, 0, 0),
              "ambient": 0.0, "diffuse": 1.0, "specular": 0.0, "shininess": 1}
    light = {"position": np.array([0, 0, -10.0]), "intensity": 1.0,
             "color": (255, 255, 255)}

    # Run intersection
    t_min, idx, normals, hit_points = intersect_spheres(
        ray_origins, ray_dirs, [sphere]
    )

    # Run shading
    image = shade_diffuse(hit_points, normals, idx, t_min, [sphere], [light])

    assert image[0, 0, 0] == 255, f"Expected R=255, got {image[0, 0, 0]}"
    assert image[0, 0, 1] == 0,   f"Expected G=0, got {image[0, 0, 1]}"
    assert image[0, 0, 2] == 0,   f"Expected B=0, got {image[0, 0, 2]}"

    return True


# ---------------------------------------------------------------------------
# Test 2: Shadow occlusion
# ---------------------------------------------------------------------------

def test_shadow_occlusion():
    """Two spheres; the second blocks light from reaching the first.

    Setup:
        - Camera/ray at (0,0,-10), direction (0,0,1)
        - Sphere A at (0,0,5), radius 1 -> front hit at (0,0,4)
          Normal at (0,0,4) = (0,0,-1)
        - Light at (0,100,4): directly above the hit point
          Light dir from hit ~ (0,1,0)  (nearly straight up)
        - Sphere B at (0,3,4), radius 2 -> directly above hit, in shadow path
          Shadow ray from ~(0,0,3.99) toward ~(0,1,0) hits B at y=1
        - Result: B blocks the shadow ray -> pixel is black

    Note: B is off-axis (+Y) so the primary ray along Z doesn't hit it.
    """
    ray_origins = np.array([[[0, 0, -10]]], dtype=np.float64)
    ray_dirs = np.array([[[0, 0, 1]]], dtype=np.float64)

    sphere_a = {"center": np.array([0, 0, 5.0]), "radius": 1.0,
                "color": (255, 0, 0),
                "ambient": 0.0, "diffuse": 1.0, "specular": 0.0, "shininess": 1}
    # Blocker is above sphere A -- the primary ray (along Z) misses it
    sphere_b = {"center": np.array([0, 3, 4.0]), "radius": 2.0,
                "color": (0, 255, 0),
                "ambient": 0.0, "diffuse": 1.0, "specular": 0.0, "shininess": 1}
    # Light is far above — shadow ray goes through B
    light = {"position": np.array([0, 100, 4.0]), "intensity": 1.0,
             "color": (255, 255, 255)}

    t_min, idx, normals, hit_points = intersect_spheres(
        ray_origins, ray_dirs, [sphere_a, sphere_b]
    )

    # Verify we hit sphere A (index 0), not B
    assert idx[0, 0] == 0, f"Expected hit on sphere A (idx 0), got {idx[0, 0]}"

    image = shade_diffuse(
        hit_points, normals, idx, t_min, [sphere_a, sphere_b], [light]
    )

    assert np.all(image[0, 0] == 0), \
        f"Expected black (shadowed), got {image[0, 0]}"

    return True


# ---------------------------------------------------------------------------
# Test 3: Multiple lights accumulate
# ---------------------------------------------------------------------------

def test_multiple_lights_accumulate():
    """Two lights illuminate the same sphere — contributions should add.

    Setup:
        - Sphere at (0,0,5), radius 1, color (255, 255, 255) — white
        - Light A at (0,0,-10), intensity 0.5, color (255, 0, 0) — red
        - Light B at (0,0,-10), intensity 0.5, color (0, 0, 255) — blue
        - Both lights hit the same front-face normal with N.L = 1.0
        - Expected: R = int(0.5*255)=127, G=0, B = int(0.5*255)=127
    """
    ray_origins = np.array([[[0, 0, -10]]], dtype=np.float64)
    ray_dirs = np.array([[[0, 0, 1]]], dtype=np.float64)

    sphere = {"center": np.array([0, 0, 5.0]), "radius": 1.0,
              "color": (255, 255, 255),
              "ambient": 0.0, "diffuse": 1.0, "specular": 0.0, "shininess": 1}
    light_a = {"position": np.array([0, 0, -10.0]), "intensity": 0.5,
               "color": (255, 0, 0)}
    light_b = {"position": np.array([0, 0, -10.0]), "intensity": 0.5,
               "color": (0, 0, 255)}

    t_min, idx, normals, hit_points = intersect_spheres(
        ray_origins, ray_dirs, [sphere]
    )

    image = shade_diffuse(
        hit_points, normals, idx, t_min, [sphere], [light_a, light_b]
    )

    # (255/255)*(255/255)*1.0*0.5 = 0.5 -> int(0.5*255) = 127
    # (255/255)*(0/255)*... = 0
    assert image[0, 0, 0] == 127, f"Expected R=127, got {image[0, 0, 0]}"
    assert image[0, 0, 1] == 0,   f"Expected G=0, got {image[0, 0, 1]}"
    assert image[0, 0, 2] == 127, f"Expected B=127, got {image[0, 0, 2]}"

    return True


# ---------------------------------------------------------------------------
# Test 4: Miss pixels are black
# ---------------------------------------------------------------------------

def test_miss_pixels_black():
    """Pixels that miss all spheres should be RGB (0, 0, 0)."""
    ray_origins = np.array([[[0, 0, 0]]], dtype=np.float64)
    ray_dirs = np.array([[[1, 0, 0]]], dtype=np.float64)  # pointing +X

    # Sphere on the +Z axis — ray goes +X, misses entirely
    sphere = {"center": np.array([0, 0, 10.0]), "radius": 1.0,
              "color": (255, 255, 255)}
    light = {"position": np.array([0, 10, 0.0]), "intensity": 1.0,
             "color": (255, 255, 255)}

    t_min, idx, normals, hit_points = intersect_spheres(
        ray_origins, ray_dirs, [sphere]
    )

    assert idx[0, 0] == -1, f"Expected miss (idx -1), got {idx[0, 0]}"

    image = shade_diffuse(
        hit_points, normals, idx, t_min, [sphere], [light]
    )

    assert np.all(image[0, 0] == 0), \
        f"Expected black for miss pixel, got {image[0, 0]}"

    return True


# ---------------------------------------------------------------------------
# Test 5: Parity with scalar trace()
# ---------------------------------------------------------------------------

def test_parity_with_scalar():
    """Compare vectorized shading output against the scalar trace() path
    on a small scene (8x8 image, 3 spheres, 2 lights).

    Tolerance: per-channel difference <= 1 (accounts for float ordering
    differences between scalar and vectorized arithmetic).
    """
    W, H = 8, 8
    cam = Camera(
        position=Vec3(0, 0, -200),
        look_at=Vec3(0, 0, 50),
        fov_deg=60,
        aspect_ratio=W / H,
    )

    # Use ambient=0, diffuse=1, specular=0 to match the scalar trace()
    # which has no material system.
    sphere_objs = [
        Sphere(Vec3(0, 0, 50), 50, (255, 0, 0),
               ambient=0.0, diffuse=1.0, specular=0.0, shininess=1),
        Sphere(Vec3(80, 0, 100), 40, (0, 255, 0),
               ambient=0.0, diffuse=1.0, specular=0.0, shininess=1),
        Sphere(Vec3(-60, -30, 120), 30, (0, 0, 255),
               ambient=0.0, diffuse=1.0, specular=0.0, shininess=1),
    ]

    light_objs = [
        Light(Vec3(100, 100, -50), 1, (255, 255, 255)),
        Light(Vec3(-100, 50, -50), 2, (100, 56, 255)),
    ]

    # --- Vectorized path ---
    ray_dirs = cam.get_rays_vectorized(W, H)
    cam_pos_np = np.array([cam.position.x, cam.position.y, cam.position.z])
    ray_origins = np.broadcast_to(cam_pos_np, ray_dirs.shape).copy()

    sphere_dicts = [sphere_to_dict(s) for s in sphere_objs]
    light_dicts = [light_to_dict(l) for l in light_objs]

    t_min, idx, normals, hit_points = intersect_spheres(
        ray_origins, ray_dirs, sphere_dicts
    )
    vec_image = shade_diffuse(
        hit_points, normals, idx, t_min, sphere_dicts, light_dicts
    )

    # --- Scalar path ---
    scalar_image = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            d = ray_dirs[y, x]
            ray_dir_v3 = Vec3(d[0], d[1], d[2])
            rgb = _scalar_trace(cam.position, ray_dir_v3, sphere_objs, light_objs)
            scalar_image[y, x] = rgb

    # --- Compare ---
    diff = np.abs(vec_image.astype(np.int16) - scalar_image.astype(np.int16))
    max_diff = np.max(diff)

    assert max_diff <= 1, (
        f"Per-channel difference too large: max_diff={max_diff}\n"
        f"  Vectorized:\n{vec_image.reshape(H*W, 3)}\n"
        f"  Scalar:\n{scalar_image.reshape(H*W, 3)}"
    )

    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("Single light, no shadows",    test_single_light_no_shadows),
    ("Shadow occlusion",            test_shadow_occlusion),
    ("Multiple lights accumulate",  test_multiple_lights_accumulate),
    ("Miss pixels are black",       test_miss_pixels_black),
    ("Parity with scalar trace()",  test_parity_with_scalar),
]


def run_tests():
    print("=" * 70)
    print("  Vectorized Diffuse Shading - Validation Suite (Week 3)")
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
