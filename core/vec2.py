# we dont use vec2 in this project but here is a simple implementation
# for the 2d testing purposes

import math

class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, o):
        if isinstance(o, tuple):
            return Vec2(self.x + o[0], self.y + o[1])
        return Vec2(self.x + o.x, self.y + o.y)

    def __radd__(self, o):
        return self.__add__(o)

    def __sub__(self, o):
        if isinstance(o, tuple):
            return Vec2(self.x - o[0], self.y - o[1])
        return Vec2(self.x - o.x, self.y - o.y)

    def __rsub__(self, o):
        if isinstance(o, tuple):
            return Vec2(o[0] - self.x, o[1] - self.y)
        return Vec2(o.x - self.x, o.y - self.y)

    def __mul__(self, s):
        return Vec2(self.x * s, self.y * s)

    def __rmul__(self, s):
        return self.__mul__(s)

    def dot(self, o):
        if isinstance(o, tuple):
            return self.x * o[0] + self.y * o[1]
        return self.x * o.x + self.y * o.y

    def length(self):
        return math.sqrt(self.dot(self))

    def normalize(self):
        l = self.length()
        return Vec2(self.x / l, self.y / l) if l != 0 else Vec2(0,0)
