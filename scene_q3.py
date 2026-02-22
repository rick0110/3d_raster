"""Scene that uses only ObjectTransform wrappers for every shape.

Notes:
- `ObjectTransform` in `src/shapes.py` accepts 3x3 linear matrices (no translation).
- This scene uses only 3x3 matrices (rotation/scale/identity). For positioning
  objects in space we rely on their constructors (e.g., `Ball(center=...)`).
"""
import math
import numpy as np
from src.base import BaseScene, Color
from src.shapes import Cube, Ball, PlaneUV, ObjectTransform, Translate, Mitchel, Heart
from src.vector3d import Vector3D
from src.camera import Camera
from src.light import PointLight, AreaLight
from src.materials import SimpleMaterialWithShadows, CheckerboardMaterial, ReflectiveMaterial

def identity3():
    return np.eye(3, dtype=float)


def rot_z(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def scale3(sx, sy, sz):
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, sz]], dtype=float)


class Scene(BaseScene):
    def __init__(self):
        super().__init__("Only ObjectTransform Scene")
        self.background = Color(0.7, 0.8, 0.95)
        self.ambient_light = Color(0.1, 0.1, 0.1)
        self.max_depth = 4
        self.camera = Camera(
            eye=Vector3D(12, 12, 3),
            look_at=Vector3D(0, 0, 2.5),
            up=Vector3D(0, 0, 1),
            fov=45,
            img_width=100,
            img_height=100,
        )

        # single point light
        self.lights = [AreaLight(
            position=Vector3D(10, 5, 15),
            look_at=Vector3D(0, 0, 0),
            up=Vector3D(1, 1, 1),
            width=4,
            height=4,
            color=Color(0.8, 0.6, 0.6),
            intensity=1.7)]

        checker = CheckerboardMaterial(1, 0.9, 1.0)
        # Azul sólido
        mat_blue = SimpleMaterialWithShadows(
            0.1, 0.8, Color(0.0, 0.0, 0.9),   # Azul sólido
            0.25, Color(1.0, 1.0, 1.0), 32,  # Brilho especular suave
        )

        red = SimpleMaterialWithShadows(0.12, 0.9, Color(0.5, 0.1, 0.1), 0.15, Color(1,1,1))

        # ground plane (wrapped in ObjectTransform identity)
        plane = PlaneUV(point=Vector3D(0,0,0), normal=Vector3D(0,0,1), forward_direction=Vector3D(1,0,0))
        self.add(ObjectTransform(plane, identity3()), checker)

        mitchel = Mitchel(n_splits_search=100, depth_bissect_search=10)
        mitchel_transformed = ObjectTransform(mitchel, rot_z(math.radians(30)) @ scale3(1.5, 1.5, 1.5))
        mitchel_transformed_translated = Translate(mitchel_transformed, Vector3D(5, 5, 3))
        self.add(mitchel_transformed_translated, mat_blue)

        heart = Heart(n_splits_search=150, depth_bissect_search=20)
        heart_transformed = ObjectTransform(heart, scale3(2, 2, 2))  
        heart_transformed_translated = Translate(heart_transformed, Vector3D(-7, -7, 3))
        self.add(heart_transformed_translated, red)

        
        