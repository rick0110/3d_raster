"""Cena mista com cilindros, bolas e cubos – alguns transformados/transladados,
outros sem transformação alguma.  Vários objetos se intersectam e usam materiais
diferentes (difuso, especular, translúcido, checkerboard).
"""
import math
import numpy as np

from src.base import BaseScene, Color
from src.shapes import Ball, Cube, Cilinder, PlaneUV, Translate, ObjectTransform
from src.camera import Camera
from src.vector3d import Vector3D
from src.light import PointLight
from src.materials import (
    SimpleMaterial,
    SimpleMaterialWithShadows,
    TranslucidMaterial,
    CheckerboardMaterial,
)


# ── helpers de transformação 3×3 ──────────────────────────────────

def identity3():
    return np.eye(3, dtype=float)


def rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rot_x(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def scale3(sx, sy, sz):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]], dtype=float)


# ── cena ──────────────────────────────────────────────────────────

class Scene(BaseScene):
    def __init__(self):
        super().__init__("Mixed Scene – Cylinders, Balls & Cubes")

        self.background = Color(0.65, 0.78, 1.0)
        self.ambient_light = Color(0.1, 0.1, 0.1)
        self.max_depth = 5

        self.camera = Camera(
            eye=Vector3D(8, -8, 6),
            look_at=Vector3D(0, 0, 1),
            up=Vector3D(0, 0, 1),
            fov=42,
            img_width=100,
            img_height=100,
        )

        self.lights = [
            PointLight(position=Vector3D(8, -6, 10), color=Color(1, 1, 1), intensity=1.4),
            PointLight(position=Vector3D(-4, 4, 8), color=Color(0.6, 0.6, 0.8), intensity=0.8),
        ]

        # ── materiais ────────────────────────────────────────────

        mat_red = SimpleMaterialWithShadows(
            0.1, 0.85, Color(0.85, 0.15, 0.15), 0.25, Color(1, 1, 1)
        )
        mat_green = SimpleMaterialWithShadows(
            0.1, 0.8, Color(0.15, 0.75, 0.2), 0.2, Color(1, 1, 1)
        )
        mat_blue = SimpleMaterial(
            ambient_coefficient=0.12,
            diffuse_coefficient=0.7,
            diffuse_color=Color(0.15, 0.3, 0.9),
            specular_coefficient=0.4,
            specular_color=Color(1, 1, 1),
            specular_shininess=64,
        )
        mat_yellow = SimpleMaterial(
            ambient_coefficient=0.1,
            diffuse_coefficient=0.8,
            diffuse_color=Color(0.9, 0.85, 0.1),
            specular_coefficient=0.15,
            specular_color=Color(1, 1, 1),
            specular_shininess=24,
        )
        mat_glass = TranslucidMaterial(
            ambient_coefficient=0.05,
            diffuse_coefficient=0.15,
            diffuse_color=Color(0.6, 0.9, 0.6),
            specular_coefficient=0.2,
            specular_color=Color(1, 1, 1),
            specular_shininess=80,
            transmission_coefficient=0.75,
            refraction_index=1.45,
        )
        mat_orange = SimpleMaterialWithShadows(
            0.1, 0.85, Color(0.95, 0.55, 0.1), 0.2, Color(1, 1, 1)
        )
        mat_purple = SimpleMaterial(
            ambient_coefficient=0.12,
            diffuse_coefficient=0.75,
            diffuse_color=Color(0.6, 0.15, 0.7),
            specular_coefficient=0.3,
            specular_color=Color(1, 1, 1),
            specular_shininess=48,
        )
        mat_checker = CheckerboardMaterial(
            ambient_coefficient=1,
            diffuse_coefficient=0.85,
            square_size=1.0,
        )

        # ── chão (plano em z=0, sem transformação) ───────────────
        self.add(
            PlaneUV(
                point=Vector3D(0, 0, 0),
                normal=Vector3D(0, 0, 1),
                forward_direction=Vector3D(1, 0, 0),
            ),
            mat_checker,
        )

        # ─────────────────────────────────────────────────────────
        # 1.  BOLA vermelha SEM transformação (centro do mundo)
        # ─────────────────────────────────────────────────────────
        self.add(Ball(center=Vector3D(0, 0, 1.2), radius=1.2), mat_red)

        # ─────────────────────────────────────────────────────────
        # 2.  CUBO azul TRANSLADADO, parcialmente dentro da bola
        # ─────────────────────────────────────────────────────────
        self.add(
            Translate(Cube(size=1.4), offset=Vector3D(1.0, 0.6, 0.7)),
            mat_blue,
        )

        # ─────────────────────────────────────────────────────────
        # 3.  CILINDRO verde SEM transformação, ao lado da bola
        # ─────────────────────────────────────────────────────────
        self.add(Cilinder(radius=0.6, height=3.0), mat_green)
        # este cilindro fica na origem e vai de z=-1.5 a z=1.5

        # ─────────────────────────────────────────────────────────
        # 4.  CILINDRO amarelo TRANSLADADO, intersectando o cubo
        # ─────────────────────────────────────────────────────────
        self.add(
            Translate(Cilinder(radius=0.45, height=2.5), offset=Vector3D(1.5, 0.6, 0)),
            mat_yellow,
        )

        # ─────────────────────────────────────────────────────────
        # 5.  BOLA de vidro (translúcida) TRANSLADADA
        # ─────────────────────────────────────────────────────────
        self.add(
            Translate(Ball(center=Vector3D(0, 0, 0), radius=0.9),
                      offset=Vector3D(-2.0, -1.5, 0.9)),
            mat_glass,
        )

        # ─────────────────────────────────────────────────────────
        # 6.  CUBO laranja ROTACIONADO + ESCALADO via ObjectTransform
        # ─────────────────────────────────────────────────────────
        m_cube_rot = rot_z(math.radians(35)) @ scale3(1.2, 0.8, 1.5)
        self.add(
            Translate(
                ObjectTransform(Cube(size=1.0), m_cube_rot),
                offset=Vector3D(-2.5, 2.0, 0.75),
            ),
            mat_orange,
        )

        # ─────────────────────────────────────────────────────────
        # 7.  CILINDRO roxo INCLINADO via ObjectTransform + transladado
        #     (intersecta a bola vermelha)
        # ─────────────────────────────────────────────────────────
        m_cyl_tilt = rot_x(math.radians(45)) @ scale3(1, 1, 1)
        self.add(
            Translate(
                ObjectTransform(Cilinder(radius=0.4, height=3.0), m_cyl_tilt),
                offset=Vector3D(0, -1.0, 1.5),
            ),
            mat_purple,
        )

        # ─────────────────────────────────────────────────────────
        # 8.  CUBO verde SEM transformação, mais afastado
        # ─────────────────────────────────────────────────────────
        self.add(
            Translate(Cube(size=1.0), offset=Vector3D(3.5, -2.0, 0.5)),
            mat_green,
        )

        # ─────────────────────────────────────────────────────────
        # 9.  BOLA azul, parcialmente dentro do cubo verde (#8)
        # ─────────────────────────────────────────────────────────
        self.add(
            Ball(center=Vector3D(3.5, -2.0, 1.3), radius=0.7),
            mat_blue,
        )


if __name__ == "__main__":
    s = Scene()
    print(f"Scene created.  Objects: {len(s.shapes)}")
    cx = s.camera.img_width // 2
    cy = s.camera.img_height // 2
    r = s.camera.ray(cx, cy)
    h = s.hit(r)
    print(f"Center ray hit: {h.hit}  t={h.t:.4f}" if h.hit else "Center ray: miss")
