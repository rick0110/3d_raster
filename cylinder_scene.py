import math
from src.base import BaseScene, Color
from src.shapes import Cilinder, PlaneUV
from src.camera import Camera
from src.vector3d import Vector3D
from src.light import PointLight
from src.materials import SimpleMaterial

class Scene(BaseScene):
    def __init__(self):
        super().__init__("Cylinder Scene")

        # quick low-res test settings
        self.background = Color(0.6, 0.8, 1)
        self.ambient_light = Color(0.2, 0.2, 0.1)
        self.max_depth = 3
        self.camera = Camera(
            eye=Vector3D(4, -4, 3.5),
            look_at=Vector3D(0, 0, 0),
            up=Vector3D(0, 0, 1),
            fov=45,
            img_width=250,
            img_height=250
        )

        self.lights = [
            PointLight(position=Vector3D(5, -5, 6), color=Color(1,1,1), intensity=1.5)
        ]

        cyl_mat = SimpleMaterial(
            ambient_coefficient=0.1,
            diffuse_coefficient=0.8,
            diffuse_color=Color(0.2, 0.6, 0.6),
            specular_coefficient=0.3,
            specular_color=Color(1,1,1),
            specular_shininess=40
        )

        ground_mat = SimpleMaterial(
            ambient_coefficient=0.2,
            diffuse_coefficient=0.8,
            diffuse_color=Color(0.9,0.9,0.9),
            specular_coefficient=0.0,
            specular_color=Color(0,0,0),
            specular_shininess=1
        )

        # Add a cylinder with base at z=0 and height=2
        self.add(Cilinder(radius=1.0, height=2.0), cyl_mat)

        # Ground plane at z=0
        self.add(PlaneUV(point=Vector3D(0,0,-1), normal=Vector3D(0,0,1), forward_direction=Vector3D(1,0,0)), ground_mat)
