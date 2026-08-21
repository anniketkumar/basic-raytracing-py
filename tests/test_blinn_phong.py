"""Validation suite for Blinn-Phong shading (Week 4).

Tests the shade_blinn_phong function: ambient illumination, specular highlights,
material property gathering, shininess control, backward compatibility, and
regression against the Week 3 diffuse parity test.

Usage:
    python -m tests.test_blinn_phong
"""

import sys
import os
import numpy as np

# Ensure the project root is on sys.path so `core.*` imports resolve.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.intersection import intersect_spheres, sphere_to_dict
from core.shading import shade_blinn_phong, shade_diffuse, light_to_dict
from core.sphere import Sphere
from core.light import Light
from core.vec3 import Vec3
from core.ray import Ray
from core.camera import Camera


# ---------------------------------------------------------------------------
# Test 1: Ambient only — no lights, ambient > 0 → pixel is not pure black
# ---------------------------------------------------------------------------

def test_ambient_only():
    """With no lights and ambient > 0, a hit pixel should NOT be pure black.

    Setup:
        - Single sphere at (0,0,5), radius 1, color (200, 100, 50)
        - Ray from (0,0,-10) toward (0,0,1) hits the sphere
        - ambient = 0.2  → pixel = (200*0.2, 100*0.2, 50*0.2) / 255 * 255
        - No lights → only ambient contributes
    """
    ray_origins = np.array([[[0, 0, -10]]], dtype=np.float64)
    ray_dirs    = np.array([[[0, 0, 1]]],   dtype=np.float64)

    sphere = {
        "center": np.array([0, 0, 5.0]), "radius": 1.0,
        "color": (200, 100, 50),
        "ambient": 0.2, "diffuse": 0.7, "specular": 0.5, "shininess": 32
    }

    t_min, idx, normals, hit_points = intersect_spheres(
        ray_origins, ray_dirs, [sphere]
    )

    # No lights — pass empty list
    image = shade_blinn_phong(
        hit_points, normals, idx, t_min, [sphere], [], ray_origins
    )

    # ambient_color = (200/255, 100/255, 50/255) * 0.2 → float
    # final = ambient_color * 255 → (200*0.2, 100*0.2, 50*0.2) = (40, 20, 10)
    expected = np.array([40, 20, 10], dtype=np.uint8)
    pixel = image[0, 0]

    assert np.allclose(pixel, expected, atol=1), \
        f"Expected ~{expected}, got {pixel}"

    # Verify it's NOT pure black
    assert np.any(pixel > 0), \
        f"Ambient-only pixel should not be pure black, got {pixel}"

    return True


# ---------------------------------------------------------------------------
# Test 2: Specular highlight — view aligned with reflection → bright specular
# ---------------------------------------------------------------------------

def test_specular_highlight():
    """When the view direction aligns with the reflection, specular is maximal.

    Setup:
        - Sphere at (0,0,5), radius 1 → front surface hit at (0,0,4)
        - Normal at (0,0,4) = (0,0,-1)
        - Camera/ray at (0,0,-10) → view_dir = normalize((0,0,-10)-(0,0,4)) = (0,0,-1)
        - Light at (0,0,-10) → light_dir = normalize((0,0,-10)-(0,0,4)) = (0,0,-1)
        - half_vec = normalize(L + V) = normalize((0,0,-2)) = (0,0,-1)
        - NdotH = dot((0,0,-1), (0,0,-1)) = 1.0
        - specular_intensity = 1.0^shininess * specular_coeff = specular_coeff
        - With specular=1.0, the specular contribution should be significant
    """
    ray_origins = np.array([[[0, 0, -10]]], dtype=np.float64)
    ray_dirs    = np.array([[[0, 0, 1]]],   dtype=np.float64)

    sphere = {
        "center": np.array([0, 0, 5.0]), "radius": 1.0,
        "color": (255, 0, 0),  # red sphere
        "ambient": 0.0, "diffuse": 0.0, "specular": 1.0, "shininess": 32
    }
    light = {
        "position": np.array([0, 0, -10.0]),
        "intensity": 1.0,
        "color": (255, 255, 255)
    }

    t_min, idx, normals, hit_points = intersect_spheres(
        ray_origins, ray_dirs, [sphere]
    )

    # WITH ray_origins → specular enabled
    image_with_spec = shade_blinn_phong(
        hit_points, normals, idx, t_min, [sphere], [light], ray_origins
    )

    # WITHOUT ray_origins → specular disabled (diffuse-only)
    image_no_spec = shade_blinn_phong(
        hit_points, normals, idx, t_min, [sphere], [light], None
    )

    # Specular uses light color (white), so all channels get contribution.
    # With diffuse=0 and ambient=0, the no-spec image should be black.
    assert np.all(image_no_spec[0, 0] == 0), \
        f"With diffuse=0, ambient=0, no-spec image should be black, got {image_no_spec[0, 0]}"

    # With spec image, NdotH=1, spec_coeff=1, so specular = 1.0 * intensity = 1.0
    # specular_contrib = light_color_01 * 1.0 * 1.0 = (1.0, 1.0, 1.0)
    # final * 255 = 255 on all channels
    assert image_with_spec[0, 0, 0] == 255, \
        f"Expected specular R=255, got {image_with_spec[0, 0, 0]}"
    assert image_with_spec[0, 0, 1] == 255, \
        f"Expected specular G=255 (light color), got {image_with_spec[0, 0, 1]}"
    assert image_with_spec[0, 0, 2] == 255, \
        f"Expected specular B=255 (light color), got {image_with_spec[0, 0, 2]}"

    return True


# ---------------------------------------------------------------------------
# Test 3: Shininess controls highlight width
# ---------------------------------------------------------------------------

def test_shininess_controls_width():
    """Higher shininess → narrower (and thus darker off-centre) specular.

    Setup:
        - Two evaluations of the same pixel with identical geometry but
          different shininess values (8 vs 128).
        - Light is slightly off-axis so NdotH < 1.0.
        - Higher shininess should produce a dimmer specular contribution
          at the same off-axis angle.

    The sphere is at (0,0,5), ray from (0,0,-10). Light at (1,0,-10) — slightly
    off-axis. This means NdotH will be slightly less than 1.0, and raising it
    to a higher power reduces the specular term.
    """
    ray_origins = np.array([[[0, 0, -10]]], dtype=np.float64)
    ray_dirs    = np.array([[[0, 0, 1]]],   dtype=np.float64)

    # Only specular, no ambient/diffuse, to isolate the shininess effect
    sphere_low = {
        "center": np.array([0, 0, 5.0]), "radius": 1.0,
        "color": (255, 255, 255),
        "ambient": 0.0, "diffuse": 0.0, "specular": 1.0, "shininess": 8
    }
    sphere_high = {
        "center": np.array([0, 0, 5.0]), "radius": 1.0,
        "color": (255, 255, 255),
        "ambient": 0.0, "diffuse": 0.0, "specular": 1.0, "shininess": 128
    }

    # Light slightly off-axis
    light = {
        "position": np.array([1, 0, -10.0]),
        "intensity": 1.0,
        "color": (255, 255, 255)
    }

    # Low shininess
    t_min, idx, normals, hit_points = intersect_spheres(
        ray_origins, ray_dirs, [sphere_low]
    )
    image_low = shade_blinn_phong(
        hit_points, normals, idx, t_min, [sphere_low], [light], ray_origins
    )

    # High shininess
    t_min, idx, normals, hit_points = intersect_spheres(
        ray_origins, ray_dirs, [sphere_high]
    )
    image_high = shade_blinn_phong(
        hit_points, normals, idx, t_min, [sphere_high], [light], ray_origins
    )

    # Low shininess should produce brighter (or equal) specular at off-axis angle
    brightness_low  = int(image_low[0, 0, 0])
    brightness_high = int(image_high[0, 0, 0])

    assert brightness_low >= brightness_high, \
        f"Low shininess ({brightness_low}) should be >= high shininess ({brightness_high}) off-axis"

    return True


# ---------------------------------------------------------------------------
# Test 4: Material defaults
# ---------------------------------------------------------------------------

def test_material_defaults():
    """Sphere with no explicit material args uses defaults (0.1, 0.7, 0.5, 32)."""
    s = Sphere(Vec3(0, 0, 0), 1.0, (255, 128, 64))

    assert s.ambient   == 0.1,  f"Default ambient should be 0.1, got {s.ambient}"
    assert s.diffuse   == 0.7,  f"Default diffuse should be 0.7, got {s.diffuse}"
    assert s.specular  == 0.5,  f"Default specular should be 0.5, got {s.specular}"
    assert s.shininess == 32,   f"Default shininess should be 32, got {s.shininess}"

    # Also verify sphere_to_dict propagates them
    d = sphere_to_dict(s)
    assert d["ambient"]   == 0.1,  f"Dict ambient should be 0.1, got {d['ambient']}"
    assert d["diffuse"]   == 0.7,  f"Dict diffuse should be 0.7, got {d['diffuse']}"
    assert d["specular"]  == 0.5,  f"Dict specular should be 0.5, got {d['specular']}"
    assert d["shininess"] == 32,   f"Dict shininess should be 32, got {d['shininess']}"

    return True


# ---------------------------------------------------------------------------
# Test 5: Backward compatibility — shade_diffuse alias still callable
# ---------------------------------------------------------------------------

def test_backward_compat_alias():
    """shade_diffuse should still be callable and return the same shape."""
    ray_origins = np.array([[[0, 0, -10]]], dtype=np.float64)
    ray_dirs    = np.array([[[0, 0, 1]]],   dtype=np.float64)

    sphere = {
        "center": np.array([0, 0, 5.0]), "radius": 1.0,
        "color": (255, 0, 0),
        "ambient": 0.1, "diffuse": 0.7, "specular": 0.5, "shininess": 32
    }
    light = {
        "position": np.array([0, 0, -10.0]),
        "intensity": 1.0,
        "color": (255, 255, 255)
    }

    t_min, idx, normals, hit_points = intersect_spheres(
        ray_origins, ray_dirs, [sphere]
    )

    # shade_diffuse should still work (backward compat)
    image = shade_diffuse(
        hit_points, normals, idx, t_min, [sphere], [light]
    )

    assert image.shape == (1, 1, 3), \
        f"Expected shape (1, 1, 3), got {image.shape}"
    assert image.dtype == np.uint8, \
        f"Expected dtype uint8, got {image.dtype}"

    # Should produce a non-black result (red sphere, white light, N.L=1)
    assert image[0, 0, 0] > 0, \
        f"Expected non-zero R channel, got {image[0, 0, 0]}"

    return True


# ---------------------------------------------------------------------------
# Test 6: Parity regression — Week 3 diffuse parity test still passes
# ---------------------------------------------------------------------------

def test_parity_regression():
    """The Week 3 parity test (vectorized vs scalar) must still pass.

    We run shade_blinn_phong on a small scene and compare against the scalar
    trace() logic.  Since the scalar trace() only computes diffuse, we set
    ambient=0 and specular=0 on the spheres to match, and use ray_origins=None
    to disable specular in the vectorized path.
    """
    W, H = 8, 8
    cam = Camera(
        position=Vec3(0, 0, -200),
        look_at=Vec3(0, 0, 50),
        fov_deg=60,
        aspect_ratio=W / H,
    )

    # Spheres with ambient=0, specular=0 to match pure diffuse
    sphere_objs = [
        Sphere(Vec3(0, 0, 50),    50, (255, 0, 0),   ambient=0, diffuse=1.0, specular=0, shininess=1),
        Sphere(Vec3(80, 0, 100),  40, (0, 255, 0),   ambient=0, diffuse=1.0, specular=0, shininess=1),
        Sphere(Vec3(-60, -30, 120), 30, (0, 0, 255), ambient=0, diffuse=1.0, specular=0, shininess=1),
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
    # Use ray_origins=None to disable specular (parity with scalar diffuse-only)
    vec_image = shade_blinn_phong(
        hit_points, normals, idx, t_min, sphere_dicts, light_dicts, None
    )

    # --- Scalar path (from test_shading.py) ---
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


def _scalar_trace(ray_origin_v3, ray_dir_v3, sphere_objs, light_objs):
    """Replicate the scalar trace() logic from viewer.py for parity testing.

    Returns:
        tuple (R, G, B) in 0-255, matching DemoViewer.trace() output.
    """
    ray = Ray(ray_origin_v3, ray_dir_v3)

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
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("Ambient only",                 test_ambient_only),
    ("Specular highlight",           test_specular_highlight),
    ("Shininess controls width",     test_shininess_controls_width),
    ("Material defaults",            test_material_defaults),
    ("Backward compat (alias)",      test_backward_compat_alias),
    ("Parity regression (diffuse)",  test_parity_regression),
]


def run_tests():
    print("=" * 70)
    print("  Blinn-Phong Shading - Validation Suite (Week 4)")
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
