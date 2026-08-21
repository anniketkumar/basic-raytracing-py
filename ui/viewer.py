import pygame
import numpy as np
from core.camera import Camera
from core.vec3 import Vec3
from core.ray import Ray
from core.sphere import Sphere
from core.light import Light
from core.intersection import intersect_spheres, sphere_to_dict
from core.shading import shade_blinn_phong, light_to_dict
#import math
#import random

class DemoViewer:
    def __init__(self, w=400, h=300):
        pygame.init()
        self.w = w
        self.h = h
        self.screen = pygame.display.set_mode((w,h))
        pygame.display.set_caption("3D Forward Ray Tracing Demo")
        self.clock = pygame.time.Clock()
        self.shapes = []
        self.lights = []
        # self.camera = Vec3(0,0,-200)
        self.camera = Camera(
            position=Vec3(0,0,-200),
            look_at=Vec3(0,0,50),
            fov_deg=60,
            aspect_ratio=self.w/self.h
        )

        self.running = True

    def create_sphere(self, pos, r, color):
        self.shapes.append(Sphere(pos, r, color))

    def create_light(self, pos, intensity, color):
        self.lights.append(Light(pos, intensity, color))

    def trace(self, ray, depth=0):
        """Trace a single ray against the scene (legacy per-pixel path).

        Kept for debugging and reference.  The main render loop now uses
        the fully vectorized pipeline in render() instead.
        """
        closest = None
        min_t = float('inf')
        for s in self.shapes:
            hit = s.intersect(ray)
            if hit and hit[0] < min_t:
                min_t = hit[0]
                closest = (s, hit)
        if not closest:
            return (0,0,0)
        s, (t, hit_point, normal) = closest
        color = [0,0,0]
        for light in self.lights:
            ldir = (light.position - hit_point).normalize()
            # shadow ray
            shadow_ray = Ray(hit_point + normal*0.01, ldir)
            blocked = False
            for o in self.shapes:
                if o != s and o.intersect(shadow_ray):
                    blocked = True
                    break
            if not blocked:
                diff = max(normal.dot(ldir),0)*light.intensity
                for i in range(3):
                    color[i] += (s.color[i] / 255) * (light.color[i] / 255) * diff
        return tuple(min(255,int(c*255)) for c in color)

    def render(self):
        """Render the full scene using the vectorized pipeline.

        End-to-end flow with zero Python-level pixel loops:
            1. Week 1 — vectorized ray direction generation  -> (H, W, 3)
            2. Week 2 — vectorized sphere intersection       -> t, idx, normals, hits
            3. Week 3 — vectorized Lambertian diffuse shading -> (H, W, 3) uint8
            4. Blit the entire image to the pygame surface in one call.
        """
        H, W = self.h, self.w

        # Guard: empty scene -> black screen
        if not self.shapes:
            self.screen.fill((0, 0, 0))
            pygame.display.flip()
            return

        # --- Week 1: vectorized ray directions ----------------------------
        ray_dirs = self.camera.get_rays_vectorized(W, H)        # (H, W, 3)

        # Build (H, W, 3) ray origins by broadcasting the camera position
        cam_pos = np.array([self.camera.position.x,
                            self.camera.position.y,
                            self.camera.position.z], dtype=np.float64)
        ray_origins = np.broadcast_to(cam_pos, ray_dirs.shape).copy()

        # Convert scene objects to dict format for the vectorized kernels
        sphere_dicts = [sphere_to_dict(s) for s in self.shapes]
        light_dicts  = [light_to_dict(l)  for l in self.lights]

        # --- Week 2: vectorized intersection ------------------------------
        t_min, idx, normals, hit_points = intersect_spheres(
            ray_origins, ray_dirs, sphere_dicts
        )

        # --- Week 4: vectorized Blinn-Phong shading ------------------------
        image = shade_blinn_phong(
            hit_points, normals, idx, t_min,
            sphere_dicts, light_dicts, ray_origins
        )

        # --- Blit to screen -----------------------------------------------
        # pygame.surfarray expects shape (W, H, 3) — axes transposed from
        # our (H, W, 3) image layout.
        pygame.surfarray.blit_array(self.screen, image.transpose(1, 0, 2))
        pygame.display.flip()

    def run(self):
        while self.running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_w:
                        self.camera.position.z += 10
                    elif e.key == pygame.K_s:
                        self.camera.position.z -= 10
                    elif e.key == pygame.K_a:
                        self.camera.position.x -= 10
                    elif e.key == pygame.K_d:
                        self.camera.position.x += 10
                    elif e.key == pygame.K_q:
                        self.camera.position.y -= 10
                    elif e.key == pygame.K_e:
                        self.camera.position.y += 10
                    elif e.key == pygame.K_UP:
                        self.camera.forward.y += 0.1
                    elif e.key == pygame.K_DOWN:
                        self.camera.forward.y -= 0.1
                    elif e.key == pygame.K_LEFT:
                        self.camera.forward.x -= 0.1
                    elif e.key == pygame.K_RIGHT:
                        self.camera.forward.x += 0.1
                    self.camera.forward = self.camera.forward.normalize()


            self.render()
            self.clock.tick(10)
        pygame.quit()
