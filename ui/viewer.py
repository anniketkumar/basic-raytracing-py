import pygame
from core.camera import Camera
from core.vec3 import Vec3
from core.ray import Ray
from core.sphere import Sphere
from core.light import Light
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
        for y in range(self.h):
            for x in range(self.w):
                #dx = (x - self.w/2)
                #dy = (y - self.h/2)
                #dz = 1
                #dir = Vec3(dx, dy, dz).normalize()
                #ray_dir = self.camera.get_ray(x, y, self.w, self.h)
                ray_dir = self.camera.get_ray(x, y, self.w, self.h)
                ray = Ray(self.camera.position, ray_dir)
                #ray = Ray(self.camera, dir)
                self.screen.set_at((x,y), self.trace(ray))
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
