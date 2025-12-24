class Ray:
    def __init__(self, origin, direction):
        self.origin = origin  # Vec3
        self.direction = direction.normalize()  # Vec3
