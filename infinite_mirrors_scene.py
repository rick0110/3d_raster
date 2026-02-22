import numpy as np
from src.base import BaseScene, Color
from src.shapes import Cube, Ball, PlaneUV, Translate
from src.vector3d import Vector3D
from src.camera import Camera
from src.light import PointLight, AreaLight
from src.materials import SimpleMaterialWithShadows, CheckerboardMaterial, ReflectiveMaterial

class Scene(BaseScene):
    def __init__(self):
        super().__init__("Infinite Mirrors")
        self.background = Color(0.7, 0.8, 0.95)
        self.ambient_light = Color(0.1, 0.1, 0.1)
        self.max_depth = 10
        
        # Câmera entre os dois espelhos, olhando levemente na diagonal para ver as múltiplas reflexões
        self.camera = Camera(
            eye=Vector3D(0.5, -2.5, 1.2),
            look_at=Vector3D(-0.2, 3.0, 1.0),
            up=Vector3D(0, 0, 1),
            fov=60,
            img_width=100,
            img_height=100,
        )

        self.lights = [
            AreaLight(
                position=Vector3D(0.0, 0.0, 3.8),
                look_at=Vector3D(0.0, 0.0, 1.0),
                up=Vector3D(0, 1, 0),
                width=1.8,
                height=1.8,
                color=Color(1.0, 1.0, 1.0),
                intensity=2.2,
            ),
            PointLight(
                position=Vector3D(-0.8, -1.8, 1.4),
                color=Color(0.65, 0.75, 1.0),
                intensity=0.6,
            ),
        ]

        # Materiais
        red = SimpleMaterialWithShadows(0.10, 0.85, Color(0.9, 0.22, 0.22), 0.25, Color(1, 1, 1), 48)
        blue = SimpleMaterialWithShadows(0.10, 0.85, Color(0.2, 0.45, 0.98), 0.22, Color(1, 1, 1), 64)
        checker = CheckerboardMaterial(
            1,
            0.85,
            0.6,
            white_color=Color(0.95, 0.95, 0.95),
            black_color=Color(0.08, 0.08, 0.09),
        )
        mirror = ReflectiveMaterial(
            0.0,
            0.05,
            Color(0.02, 0.02, 0.02),
            0.9,
            Color(1, 1, 1),
            220,
            reflection_coefficient=0.95,
        )

        # Chão
        floor = PlaneUV(
            point=Vector3D(0, 0, 0),
            normal=Vector3D(0, 0, 1),
            forward_direction=Vector3D(0, 1, 0),
        )
        self.add(floor, checker)

        # Dois espelhos frente a frente
        mirror_y = 3.0
        front_mirror = PlaneUV(
            point=Vector3D(0, mirror_y, 0),
            normal=Vector3D(0, -1, 0),
            forward_direction=Vector3D(0, 0, 1),
        )
        back_mirror = PlaneUV(
            point=Vector3D(0, -mirror_y, 0),
            normal=Vector3D(1, 1, 0),
            forward_direction=Vector3D(0, 0, 1),
        )
        self.add(front_mirror, mirror)
        self.add(back_mirror, mirror)

        # Duas bolas sólidas
        self.add(Ball(center=Vector3D(-1.0, 0.0, 0.5), radius=0.5), red)
        self.add(Ball(center=Vector3D(1.2, 1.5, 0.6), radius=0.6), blue)

        # Um quadrado (cubo) espelhado
        self.add(Translate(Cube(size=0.8), Vector3D(0.0, 0.5, 0.4)), mirror)
