import numpy as np
from src.base import BaseScene, Color
from src.shapes import Cube, Ball, PlaneUV, ObjectTransform, Translate, Mitchel, Heart
from src.vector3d import Vector3D
from src.camera import Camera
from src.light import PointLight, AreaLight
from src.materials import SimpleMaterialWithShadows, CheckerboardMaterial, ReflectiveMaterial


class Scene(BaseScene):
    def __init__(self):
        super().__init__("Infinite Mirrors (camera between)")
        self.background = Color(0.03, 0.03, 0.04)
        self.ambient_light = Color(0.06, 0.06, 0.06)
        self.max_depth = 60
        self.camera = Camera(
            # câmera entre dois espelhos paralelos (x = ±mirror_x)
            eye=Vector3D(0.0, -2.2, 1.25),
            look_at=Vector3D(2.0, 0.15, 1.05),
            up=Vector3D(0, 0, 1),
            fov=55,
            img_width=250,
            img_height=250,
        )

        self.lights = [
            AreaLight(
                position=Vector3D(0.0, -0.6, 3.8),
                look_at=Vector3D(0.0, 2.0, 1.0),
                up=Vector3D(0, 0, 1),
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

        # materials
        red = SimpleMaterialWithShadows(0.10, 0.85, Color(0.9, 0.22, 0.22), 0.25, Color(1, 1, 1), 48)
        green = SimpleMaterialWithShadows(0.10, 0.85, Color(0.22, 0.9, 0.3), 0.20, Color(1, 1, 1), 48)
        blue = SimpleMaterialWithShadows(0.10, 0.85, Color(0.2, 0.45, 0.98), 0.22, Color(1, 1, 1), 64)
        gold = SimpleMaterialWithShadows(0.10, 0.85, Color(0.95, 0.78, 0.15), 0.30, Color(1, 1, 1), 96)
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
            reflection_coefficient=0.92,
        )

        # chão
        floor = PlaneUV(
            point=Vector3D(0, 0, 0),
            normal=Vector3D(0, 0, 1),
            forward_direction=Vector3D(0, 1, 0),
        )
        self.add(floor, checker)

        # dois espelhos frente a frente (planos infinitos)
        mirror_x = 2.0
        left_mirror = PlaneUV(
            point=Vector3D(-mirror_x, 0, 0),
            normal=Vector3D(1, 0, 0),
            forward_direction=Vector3D(0, 0, 1),
        )
        right_mirror = PlaneUV(
            point=Vector3D(mirror_x, 0, 0),
            normal=Vector3D(-1, 0, 0),
            forward_direction=Vector3D(0, 0, 1),
        )
        self.add(left_mirror, mirror)
        self.add(right_mirror, mirror)

        # objetos para “marcar” as múltiplas reflexões
        self.add(Ball(center=Vector3D(-0.9, 0.8, 0.55), radius=0.55), red)
        self.add(Ball(center=Vector3D(0.75, 1.55, 0.35), radius=0.35), green)
        self.add(Ball(center=Vector3D(0.15, 2.6, 0.7), radius=0.7), blue)
        self.add(Translate(Cube(size=0.65), Vector3D(-0.15, 1.9, 0.325)), gold)

        # um objeto “escultural” no eixo para ficar bem evidente nas reflexões
        heart_small = ObjectTransform(Heart(n_splits_search=55, depth_bissect_search=28), np.eye(3) * 0.55)
        self.add(Translate(heart_small, Vector3D(0.25, 3.8, 0.65)), red)

        # e um Mitchel menor para variar as formas
        mitchel_small = ObjectTransform(Mitchel(n_splits_search=55, depth_bissect_search=28), np.eye(3) * 0.45)
        self.add(Translate(mitchel_small, Vector3D(-0.55, 4.8, 0.55)), blue)

        



        