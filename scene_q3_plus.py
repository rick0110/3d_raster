"""Cena criativa 'Jardim Encantado' – usa TODAS as formas disponíveis:
Ball, Cube, Cilinder, Heart, Mitchel, Paraboloid, PlaneUV.
Transformações: ObjectTransform (rotação/escala) e Translate.
Materiais: SimpleMaterial, SimpleMaterialWithShadows, TranslucidMaterial,
           CheckerboardMaterial.
Luzes: PointLight + AreaLight.
Sem espelhos (ReflectiveMaterial).
"""
import math
import numpy as np

from src.base import BaseScene, Color
from src.shapes import (
    Ball, Cube, Cilinder, PlaneUV,
    Heart, Mitchel, Paraboloid,
    ObjectTransform, Translate,
)
from src.vector3d import Vector3D
from src.camera import Camera
from src.light import PointLight, AreaLight
from src.materials import (
    SimpleMaterial,
    SimpleMaterialWithShadows,
    TranslucidMaterial,
    CheckerboardMaterial,
)


# ── helpers de transformação 3×3 ──────────────────────────────────

def rot_x(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

def rot_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

def rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

def scale3(sx, sy, sz):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]], dtype=float)

def identity3():
    return np.eye(3, dtype=float)


# ── cena ──────────────────────────────────────────────────────────

class Scene(BaseScene):
    def __init__(self):
        super().__init__("Jardim Encantado – All Shapes Showcase")

        self.background = Color(0.15, 0.05, 0.25)   # céu noturno roxo
        self.ambient_light = Color(0.08, 0.06, 0.12)
        self.max_depth = 5

        # câmera panorâmica – vista ampla do jardim
        self.camera = Camera(
            eye=Vector3D(10, -10, 8),
            look_at=Vector3D(0, 0, 2),
            up=Vector3D(0, 0, 1),
            fov=50,
            img_width=700,
            img_height=700,
        )

        # ── iluminação ───────────────────────────────────────────
        # Luz principal quente (area light para sombras suaves)
        self.lights = [
            AreaLight(
                position=Vector3D(10, -6, 14),
                look_at=Vector3D(0, 0, 0),
                up=Vector3D(0, 0, 1),
                width=5, height=5,
                color=Color(1.0, 0.9, 0.7),
                intensity=1.5,
            ),
            # Luz de preenchimento fria (lateral)
            PointLight(
                position=Vector3D(-8, 6, 10),
                color=Color(0.4, 0.5, 1.0),
                intensity=0.7,
            ),
            # Luz de baixo, toque dramático
            PointLight(
                position=Vector3D(0, 0, 0.3),
                color=Color(0.9, 0.3, 0.6),
                intensity=0.4,
            ),
        ]

        # ── materiais ────────────────────────────────────────────

        # Chão xadrez roxo/branco
        mat_checker = CheckerboardMaterial(
            ambient_coefficient=1.0,
            diffuse_coefficient=0.8,
            square_size=1.5,
            white_color=Color(0.85, 0.8, 0.95),
            black_color=Color(0.2, 0.1, 0.3),
        )

        # Vermelho rubi vibrante
        mat_ruby = SimpleMaterialWithShadows(
            0.1, 0.85, Color(0.85, 0.05, 0.15),
            0.4, Color(1, 1, 1), 64,
        )
    
        # Azul profundo
        mat_deep_blue = SimpleMaterialWithShadows(
            0.1, 0.8, Color(0.08, 0.15, 0.85),
            0.3, Color(0.8, 0.8, 1.0), 48,
        )

        # Dourado brilhante
        mat_gold = SimpleMaterial(
            ambient_coefficient=0.12,
            diffuse_coefficient=0.7,
            diffuse_color=Color(0.95, 0.75, 0.1),
            specular_coefficient=0.5,
            specular_color=Color(1, 1, 0.8),
            specular_shininess=80,
        )

        # Verde esmeralda fosco
        mat_emerald = SimpleMaterialWithShadows(
            0.1, 0.85, Color(0.05, 0.7, 0.3),
            0.2, Color(0.5, 1, 0.5), 32,
        )

        # Roxo místico
        mat_mystic = SimpleMaterialWithShadows(
            0.12, 0.75, Color(0.55, 0.1, 0.8),
            0.35, Color(1, 0.7, 1), 56,
        )

        # Vidro translúcido azulado
        mat_glass_blue = TranslucidMaterial(
            ambient_coefficient=0.03,
            diffuse_coefficient=0.1,
            diffuse_color=Color(0.3, 0.5, 0.9),
            specular_coefficient=0.3,
            specular_color=Color(1, 1, 1),
            specular_shininess=100,
            transmission_coefficient=0.8,
            refraction_index=1.5,
        )

        # Vidro translúcido rosado
        mat_glass_pink = TranslucidMaterial(
            ambient_coefficient=0.03,
            diffuse_coefficient=0.12,
            diffuse_color=Color(0.9, 0.3, 0.5),
            specular_coefficient=0.25,
            specular_color=Color(1, 1, 1),
            specular_shininess=90,
            transmission_coefficient=0.75,
            refraction_index=1.4,
        )

        # Laranja quente
        mat_orange = SimpleMaterialWithShadows(
            0.1, 0.85, Color(0.95, 0.5, 0.05),
            0.25, Color(1, 1, 0.8), 40,
        )

        # Branco perolado
        mat_pearl = SimpleMaterial(
            ambient_coefficient=0.15,
            diffuse_coefficient=0.7,
            diffuse_color=Color(0.95, 0.92, 0.88),
            specular_coefficient=0.45,
            specular_color=Color(1, 1, 1),
            specular_shininess=100,
        )

        # ═════════════════════════════════════════════════════════
        # OBJETOS DA CENA
        # ═════════════════════════════════════════════════════════

        # ── 1. CHÃO – PlaneUV com checkerboard ──────────────────
        self.add(
            PlaneUV(
                point=Vector3D(0, 0, 0),
                normal=Vector3D(0, 0, 1),
                forward_direction=Vector3D(1, 0, 0),
            ),
            mat_checker,
        )

        # ── 2. CORAÇÃO translúcido flutuando ────────────────────
        #    Rotacionado para ficar "de pé" e escalado
        heart = Heart()
        heart_xform = ObjectTransform(
            heart,
            rot_x(math.radians(90)) @ scale3(1.8, 1.8, 1.8),
        )
        self.add(
            Translate(heart_xform, Vector3D(0, 0, 4.5)),
            mat_glass_pink,
        )

        # ── 3. MITCHEL – superfície implícita mística ───────────
        #    Rotacionado levemente e posicionado ao lado
        mitchel = Mitchel()
        mitchel_xform = ObjectTransform(
            mitchel,
            rot_z(math.radians(45)) @ scale3(1.5, 1.5, 1.5),
        )
        self.add(
            Translate(mitchel_xform, Vector3D(-4, 2, 3)),
            mat_mystic,
        )

        # ── 4. PARABOLOIDE – "taça" dourada ─────────────────────
        #    Paraboloide virado para cima, como um cálice
        para = Paraboloid(k=0.3)
        para_xform = ObjectTransform(para, scale3(2.5, 2.5, 2.5))
        self.add(
            Translate(para_xform, Vector3D(4, 3, 0.5)),
            mat_gold,
        )

        # ── 5. ESFERA de vidro azul dentro do paraboloide ───────
        self.add(
            Ball(center=Vector3D(4, 3, 2.0), radius=0.8),
            mat_glass_blue,
        )

        # ── 6. CUBO dourado – pedestal rotacionado ──────────────
        cube_rot = ObjectTransform(
            Cube(size=1.6),
            rot_z(math.radians(30)) @ scale3(1, 1, 1.5),
        )
        self.add(
            Translate(cube_rot, Vector3D(0, 0, 1.2)),
            mat_gold,
        )

        # ── 7. CILINDRO pilar esmeralda (esquerda) ──────────────
        self.add(
            Translate(
                Cilinder(radius=0.5, height=5.0),
                Vector3D(-5, -2, 2.5),
            ),
            mat_emerald,
        )

        # ── 8. CILINDRO pilar azul (direita), inclinado ─────────
        cyl_tilt = ObjectTransform(
            Cilinder(radius=0.4, height=4.5),
            rot_x(math.radians(15)) @ rot_z(math.radians(-10)),
        )
        self.add(
            Translate(cyl_tilt, Vector3D(5, -3, 2.2)),
            mat_deep_blue,
        )

        # ── 9. ESFERA RUBI grande – protagonista ────────────────
        self.add(
            Ball(center=Vector3D(-2, -3, 1.5), radius=1.5),
            mat_ruby,
        )

        # ── 10. CUBO translúcido azul, parcialmente na esfera ──
        cube_glass = ObjectTransform(
            Cube(size=1.2),
            rot_z(math.radians(45)) @ rot_x(math.radians(20)),
        )
        self.add(
            Translate(cube_glass, Vector3D(-1.2, -3, 2.0)),
            mat_glass_blue,
        )

        # ── 11. ESFERA perolada pequena flutuante ───────────────
        self.add(
            Ball(center=Vector3D(2.5, -1.5, 5.5), radius=0.6),
            mat_pearl,
        )

        # ── 12. ESFERA laranja no chão ──────────────────────────
        self.add(
            Ball(center=Vector3D(6, 0, 0.7), radius=0.7),
            mat_orange,
        )

        # ── 13. CILINDRO laranja deitado – tronco caído ────────
        cyl_lying = ObjectTransform(
            Cilinder(radius=0.35, height=4.0),
            rot_y(math.radians(90)),
        )
        self.add(
            Translate(cyl_lying, Vector3D(3, -4, 0.35)),
            mat_orange,
        )

        # ── 14. CUBO esmeralda pequeno – decoração ──────────────
        small_cube = ObjectTransform(
            Cube(size=0.8),
            rot_z(math.radians(60)),
        )
        self.add(
            Translate(small_cube, Vector3D(-5, -2, 5.4)),
            mat_emerald,
        )

        # ── 15. ESFERA mística flutuante ────────────────────────
        self.add(
            Ball(center=Vector3D(-3, 4, 3.5), radius=0.9),
            mat_mystic,
        )

        # ── 16. CUBO rubi – empilhado, levemente torto ─────────
        cube_stack = ObjectTransform(
            Cube(size=1.0),
            rot_z(math.radians(15)) @ rot_x(math.radians(10)),
        )
        self.add(
            Translate(cube_stack, Vector3D(6, 0, 1.8)),
            mat_ruby,
        )

