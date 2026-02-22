import math
import numpy as np
from src.base import BaseScene, Color
from src.shapes import Cube, Ball, PlaneUV, ObjectTransform, Paraboloid, Translate
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
        super().__init__("Only ObjectTransform Scene")
        self.background = Color(0.7, 0.8, 0.95)
        self.ambient_light = Color(0.1, 0.1, 0.1)
        self.max_depth = 4
        self.camera = Camera(
            eye=Vector3D(8, -8, 6),
            look_at=Vector3D(0, 0, 2.5),
            up=Vector3D(0, 0, 1),
            fov=45,
            img_width=700,
            img_height=700,
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

        # materials
        red = SimpleMaterialWithShadows(0.12, 0.9, Color(0.85, 0.2, 0.2), 0.2, Color(1,1,1))
        green = SimpleMaterialWithShadows(0.12, 0.9, Color(0.2, 0.85, 0.2), 0.15, Color(1,1,1))
        blue = SimpleMaterialWithShadows(0.12, 0.9, Color(0.2, 0.4, 0.95), 0.15, Color(1,1,1))
        checker = CheckerboardMaterial(1, 0.9, 1.0)
        specular = SimpleMaterialWithShadows(0.4, 0.8, Color(0.8, 0.8, 0.8), 0.7, Color(0.9,0.9,0.9))
        # ground plane (wrapped in ObjectTransform identity)
        plane = PlaneUV(point=Vector3D(0,0,0), normal=Vector3D(0,0,1), forward_direction=Vector3D(1,0,0))
        self.add(ObjectTransform(plane, identity3()), checker)

        # cube at origin but wrapped
        cube = Cube(size=1.0)
        cube_transform = rot_z(math.radians(45)) @ scale3(1.5, 1.5, 1.5)
        cube_transformed = ObjectTransform(cube, cube_transform)
        cube_transformed_translated = Translate(cube_transformed, Vector3D(0, 0, 3))  
        self.add(cube_transformed_translated, red)

        # ball (position given in center) - still wrapped with identity
        ball = Ball(center=Vector3D(-2.5, 0.0, 0.8), radius=0.8)
        ball_transformed = ObjectTransform(ball, rot_x(-math.radians(30))@rot_z(-math.radians(45))@np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 2.0]], dtype=float))
        ball_transformed_translated = Translate(ball_transformed, Vector3D(1, 1, 2))
        self.add(ball_transformed_translated, green)

        # paraboloid (position given in vertex) - still wrapped with identity
        paraboloid = Paraboloid(3)
        paraboloid_transformed = ObjectTransform(paraboloid, scale3(1.0, 1.0, 2.0) @ rot_x(-math.radians(45)) @ rot_y(-math.radians(45)) @ rot_z(math.radians(45)))
        paraboloid_transformed_translated = Translate(paraboloid_transformed, Vector3D(-1, -1, 0))
        self.add(paraboloid_transformed_translated, specular)

        self.add(Cube(size=2), blue)

        self.add(Ball(center=Vector3D(0,0, 0), radius=1.3), red)
        




