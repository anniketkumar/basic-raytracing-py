import math
class Sphere():
    def __init__(self, position, radius, color,
                 ambient=0.1, diffuse=0.7, specular=0.5, shininess=32):
        self.position = position
        self.radius = radius
        self.color = color
        self.ambient = ambient      # Blinn-Phong ambient coefficient
        self.diffuse = diffuse      # Blinn-Phong diffuse coefficient
        self.specular = specular    # Blinn-Phong specular coefficient
        self.shininess = shininess  # Specular exponent (higher = tighter highlight)

    def change_color(self, new_color):
        self.color = new_color

    def set_postion(self, new_position):
        self.position = new_position
    
    def get_info(self):
        return {
            "position": self.position,
            "radius": self.radius,
            "color": self.color
        }

    def intersect(self, ray):
        oc = ray.origin - self.position
        a = ray.direction.dot(ray.direction)
        b = 2 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius*self.radius
        disc = b*b - 4*a*c
        if disc < 0:
            return None
        t = (-b - math.sqrt(disc)) / (2*a)
        if t < 0:
            return None
        hit_point = ray.origin + ray.direction * t
        normal = (hit_point - self.position).normalize()
        return t, hit_point, normal