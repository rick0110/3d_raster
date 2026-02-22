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

def rot_x(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)

def rot_y(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)

def scale3(sx, sy, sz):
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, sz]], dtype=float)

class Scene(BaseScene):
    def __init__(self):
        super().__init__("Heart and Mitchel Scene")
        self.background = Color(0.05, 0.05, 0.1) # Dark background
        self.ambient_light = Color(0.1, 0.1, 0.15)
        self.max_depth = 4
        
        self.camera = Camera(
            eye=Vector3D(-5, 10, 6),
            look_at=Vector3D(0, 0, 2),
            up=Vector3D(0, 0, 1),
            fov=50,
            img_width=100,
            img_height=100,
        )

        # Two lights for dramatic effect
        self.lights = [
            AreaLight(
                position=Vector3D(8, -8, 10),
                look_at=Vector3D(0, 0, 2),
                up=Vector3D(0, 0, 1),
                width=3,
                height=3,
                color=Color(1.0, 0.9, 0.9),
                intensity=1.5
            ),
            PointLight(
                position=Vector3D(-5, 5, 8),
                color=Color(0.4, 0.6, 1.0),
                intensity=0.8
            )
        ]

        # Materials
        checker = CheckerboardMaterial(1.5, 0.8, 0.2)
        
        mat_heart = SimpleMaterialWithShadows(
            0.1, 0.9, Color(0.9, 0.1, 0.2),   # Bright red
            0.3, Color(1.0, 0.8, 0.8), 64,    # High specular
        )
        
        mat_mitchel = ReflectiveMaterial(
            0.1, 0.7, Color(0.2, 0.5, 0.9),   # Blue
            0.4, Color(1.0, 1.0, 1.0), 32,
            0.3 # Reflectivity
        )
        
        mat_gold = ReflectiveMaterial(
            0.2, 0.6, Color(0.8, 0.6, 0.1),
            0.5, Color(1.0, 0.9, 0.5), 64,
            0.4
        )

        # Ground plane
        plane = PlaneUV(point=Vector3D(0,0,0), normal=Vector3D(0,0,1), forward_direction=Vector3D(1,0,0))
        self.add(ObjectTransform(plane, identity3()), checker)

        # Heart shape
        heart = Heart(n_splits_search=100, depth_bissect_search=20)
        # Rotate to stand up and face camera slightly, scale up
        heart_rot = rot_x(math.radians(90)) @ rot_z(math.radians(45))
        heart_transformed = ObjectTransform(heart, heart_rot @ scale3(1.5, 1.5, 1.5))
        heart_translated = Translate(heart_transformed, Vector3D(-2, 2, 2.5))
        self.add(heart_translated, mat_heart)

        # Mitchel shape
        mitchel = Mitchel(n_splits_search=80, depth_bissect_search=15)
        mitchel_transformed = ObjectTransform(mitchel, rot_z(math.radians(-30)) @ scale3(1.2, 1.2, 1.2))
        mitchel_translated = Translate(mitchel_transformed, Vector3D(2, -2, 2))
        self.add(mitchel_translated, mat_mitchel)
        
        # Add a reflective ball between them
        ball = Ball(center=Vector3D(0, 0, 1.5), radius=1.5)
        self.add(ball, mat_gold)
