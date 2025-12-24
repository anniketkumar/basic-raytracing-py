from core.vec3 import Vec3
import math

class Camera:
    def __init__(self, position, look_at, fov_deg, aspect_ratio):
        self.position = position
        self.forward = (look_at - position).normalize()
        self.right = Vec3(0,1,0).cross(self.forward).normalize()
        self.up = self.forward.cross(self.right).normalize()
        self.fov = math.radians(fov_deg)
        self.aspect_ratio = aspect_ratio

    def get_ray(self, px, py, screen_width, screen_height):
        # normalize pixel coordinates to [-1,1]
        x = (2 * (px + 0.5)/screen_width - 1) * math.tan(self.fov/2) * self.aspect_ratio
        y = (1 - 2*(py + 0.5)/screen_height) * math.tan(self.fov/2)
        ray_dir = (self.forward + self.right*x + self.up*y).normalize()
        return ray_dir