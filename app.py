from ui.viewer import DemoViewer
from core.vec3 import Vec3

app = DemoViewer(300, 300)

# Spheres
app.create_sphere(Vec3(0,0,50), 50, (255,0,0))
app.create_sphere(Vec3(80,0,100), 40, (0,255,0))
app.create_sphere(Vec3(-60,-30,120), 30, (0,0,255))
app.create_sphere(Vec3(0,60,80), 35, (255,255,255))
app.create_sphere(Vec3(0,-70,60), 30, (255,255,255))

# Lights
app.create_light(Vec3(100,100,-50), 1, (255,255,255))
app.create_light(Vec3(-100,50,-50), 2, (100, 56, 255))
app.create_light(Vec3(0,-100,0), 4, (29, 191, 40))


app.run()
