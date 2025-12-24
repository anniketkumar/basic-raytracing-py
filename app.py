from ui.viewer import DemoViewer
from core.vec3 import Vec3

app = DemoViewer(300, 300)


def first_example():
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


def second_example():
    
    # Spheres
    app.create_sphere(Vec3(-40, -20, 70), 45, (255, 100, 150))
    app.create_sphere(Vec3(60, 30, 90), 35, (255, 200, 0))
    app.create_sphere(Vec3(-70, 50, 120), 40, (0, 200, 255))
    app.create_sphere(Vec3(0, -60, 50), 25, (200, 0, 200))
    app.create_sphere(Vec3(80, -40, 110), 30, (0, 255, 100))
    app.create_sphere(Vec3(-30, 70, 85), 28, (255, 255, 100))
    app.create_sphere(Vec3(20, 10, 140), 50, (100, 150, 255))

    # Lights
    app.create_light(Vec3(-120, 80, -40), 1.0, (255, 220, 180))
    app.create_light(Vec3(110, -70, -30), 0.3, (180, 100, 255))
    app.create_light(Vec3(0, 120, -20), 1.4, (255, 150, 50))
    app.create_light(Vec3(90, 0, 10), 1.0, (0, 255, 200))



app.run()
