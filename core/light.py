from core.ray import Ray
from core.vec3 import Vec3
class Light:

    def __init__(self, position, intensity, color):
        self.position = position  # Vec3
        self.intensity = intensity
        self.color = color  # tuple 0-255

    def set_position(self, new_position):
        self.position = new_position
    
    def set_intensity(self, new_intensity):
        self.intensity = new_intensity


    def get_info(self):
        return {
            "position": self.position,
            "intensity": self.intensity,
            "color": self.color
        }

    