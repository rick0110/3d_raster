import math
from src.base import BaseScene, Color
from src.shapes import Cube, PlaneUV
from src.camera import Camera
from src.vector3d import Vector3D
from src.light import PointLight
from src.materials import SimpleMaterial

class Scene(BaseScene):
    def __init__(self):
        super().__init__("Cube Scene")

        # low-res for quick tests
        self.background = Color(0.7, 0.8, 1)
        self.ambient_light = Color(0.1, 0.1, 0.1)
        self.max_depth = 3
        self.camera = Camera(
            eye=Vector3D(4, 4, 3),
            look_at=Vector3D(0, 0, 0),
            up=Vector3D(0, 0, 1),
            fov=45,
            img_width=100,
            img_height=100
        )

        self.lights = [
            PointLight(position=Vector3D(5, 5, 6), color=Color(1,1,1), intensity=1.6)
        ]

        mat = SimpleMaterial(
            ambient_coefficient=0.1,
            diffuse_coefficient=0.8,
            diffuse_color=Color(0.6, 0.2, 0.2),
            specular_coefficient=0.2,
            specular_color=Color(1,1,1),
            specular_shininess=32
        )

        ground_mat = SimpleMaterial(
            ambient_coefficient=0.2,
            diffuse_coefficient=0.8,
            diffuse_color=Color(0.8,0.8,0.8),
            specular_coefficient=0.0,
            specular_color=Color(0,0,0),
            specular_shininess=1
        )

        # Add cube at origin and a ground plane for reference
        self.add(Cube(size=2.0), mat)
        self.add(PlaneUV(point=Vector3D(0,0,-1), normal=Vector3D(0,0,1), forward_direction=Vector3D(1,0,0)), ground_mat)
