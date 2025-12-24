import math
import random

class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, o):
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o):
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, s):
        return Vec3(self.x * s, self.y * s, self.z * s)

    def __truediv__(self, s):
        return Vec3(self.x / s, self.y / s, self.z / s)

    def dot(self, o):
        return self.x*o.x + self.y*o.y + self.z*o.z

    def length(self):
        return math.sqrt(self.dot(self))

    def normalize(self):
        l = self.length()
        return self / l if l != 0 else Vec3(0,0,0)

    def cross(self, o):
        return Vec3(
            self.y*o.z - self.z*o.y,
            self.z*o.x - self.x*o.z,
            self.x*o.y - self.y*o.x
        )
    
    @staticmethod
    def random_unit():
        while True:
            x = random.uniform(-1,1)
            y = random.uniform(-1,1)
            z = random.uniform(-1,1)
            v = Vec3(x,y,z)
            if v.length() <= 1:
                return v.normalize()

    