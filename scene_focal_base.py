import math
from src.base import BaseScene, Color
from src.shapes import Ball, PlaneUV
from src.camera import Camera_with_focal_depth
from src.vector3d import Vector3D
from src.light import PointLight, AreaLight
from src.materials import SimpleMaterial, CheckerboardMaterial

class BaseFocalScene(BaseScene):
    def __init__(self, radius, focal_dist):
        super().__init__("Focal Depth Scene")

        self.background = Color(0.7, 0.8, 1)
        self.ambient_light = Color(0.1, 0.1, 0.1)
        self.max_depth = 5
        
        self.camera = Camera_with_focal_depth(
            eye=Vector3D(0, -15, 2),
            look_at=Vector3D(0, 5, 1),
            up=Vector3D(0, 0, 1),
            fov=45,
            img_width=500,
            img_height=500,
            radius=radius,
            focal_dist=focal_dist
        )
        
        self.lights = [
            PointLight(
                position=Vector3D(10, -10, 10),
                color=Color(1, 1, 1),
                intensity=1.0
            ),
            PointLight(
                position=Vector3D(-10, 5, 10),
                color=Color(1, 1, 1),
                intensity=0.8
            )
        ]

        colors = [
            Color(1, 0, 0),
            Color(0, 1, 0),
            Color(0, 0, 1),
            Color(1, 1, 0),
            Color(1, 0, 1)
        ]
        
        positions = [
            Vector3D(-4, -5, 1),
            Vector3D(-2, 0, 1),
            Vector3D(0, 5, 1),
            Vector3D(2, 10, 1),
            Vector3D(4, 15, 1)
        ]

        for i in range(5):
            mat = SimpleMaterial(
                ambient_coefficient=1,
                diffuse_coefficient=0.8,
                diffuse_color=colors[i],
                specular_coefficient=0.5,
                specular_color=Color(1, 1, 1),
                specular_shininess=32
            )
            self.add(Ball(center=positions[i], radius=1), mat)

        gray_material = CheckerboardMaterial(
            ambient_coefficient=1,
            diffuse_coefficient=0.8,
            square_size=2.0,
            white_color=Color(0.9, 0.9, 0.9),
            black_color=Color(0.4, 0.4, 0.4)
        )
        self.add(PlaneUV(point=Vector3D(0, 0, 0), normal=Vector3D(0, 0, 1), forward_direction=Vector3D(1, 0, 0)), gray_material)
