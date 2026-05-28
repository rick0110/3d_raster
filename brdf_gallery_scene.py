"""Wide BRDF gallery scene.

The scene samples materials from material_data/BRDFDatabase and applies them to
teapot-like composites so the render shows measured reflectance variations in a
composition similar to the reference image.
"""

from __future__ import annotations

import math

import numpy as np

from raytracer.base import BaseScene, Color
from raytracer.brdf import MerlBrdfDatabase, MerlBrdfMaterial
from raytracer.camera import Camera
from raytracer.light import AreaLight, PointLight
from raytracer.materials import CheckerboardMaterial, SimpleMaterial, SimpleMaterialWithShadows
from raytracer.shapes import Ball, Cilinder, ObjectTransform, Plane, Translate
from raytracer.vector3d import Vector3D


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


def _pick_materials(names, count):
    if not names:
        return []
    if len(names) <= count:
        return names
    indices = np.linspace(0, len(names) - 1, count, dtype=int)
    return [names[index] for index in indices]


def _add_teapot_like_group(scene, material, center, scale=1.0, yaw=0.0):
    body_transform = rot_z(yaw) @ scale3(scale, scale, scale * 0.92)
    spout_transform = rot_z(yaw) @ rot_y(math.pi / 2.0) @ scale3(scale * 0.95, scale * 0.18, scale * 0.18)
    handle_transform = rot_z(yaw) @ rot_x(math.pi / 2.0) @ scale3(scale * 0.88, scale * 0.14, scale * 0.14)

    body = Translate(
        ObjectTransform(Ball(center=Vector3D(0, 0, 0), radius=1.0), body_transform),
        center + Vector3D(0, 0, 1.0 * scale),
    )
    lid = Translate(
        ObjectTransform(Ball(center=Vector3D(0, 0, 0), radius=0.26), rot_z(yaw) @ scale3(scale * 0.9, scale * 0.9, scale * 0.35)),
        center + Vector3D(0, 0, 2.03 * scale),
    )
    knob = Translate(Ball(center=Vector3D(0, 0, 0), radius=0.1 * scale), center + Vector3D(0, 0, 2.32 * scale))
    spout = Translate(
        ObjectTransform(Cilinder(radius=0.09 * scale, height=1.15 * scale), spout_transform),
        center + Vector3D(1.12 * scale, 0.0, 0.95 * scale),
    )
    handle = Translate(
        ObjectTransform(Cilinder(radius=0.085 * scale, height=1.15 * scale), handle_transform),
        center + Vector3D(-1.02 * scale, 0.0, 1.0 * scale),
    )

    scene.add(body, material)
    scene.add(lid, material)
    scene.add(knob, material)
    scene.add(spout, material)
    scene.add(handle, material)


class Scene(BaseScene):
    def __init__(self):
        super().__init__("BRDF Gallery Scene")

        self.background = Color(0.74, 0.72, 0.68)
        self.ambient_light = Color(0.08, 0.08, 0.08)
        self.max_depth = 2
        self.camera = Camera(
            eye=Vector3D(0.0, -17.0, 7.2),
            look_at=Vector3D(0.0, 0.7, 1.7),
            up=Vector3D(0, 0, 1),
            fov=36,
            img_width=1500,
            img_height=460,
        )

        self.lights = [
            AreaLight(
                position=Vector3D(0.0, -4.0, 13.0),
                look_at=Vector3D(0.0, 2.0, 0.5),
                up=Vector3D(0, 0, 1),
                width=8.0,
                height=6.0,
                color=Color(1.0, 0.97, 0.92),
                intensity=1.8,
            ),
            PointLight(
                position=Vector3D(-8.0, -10.0, 6.0),
                color=Color(0.65, 0.72, 1.0),
                intensity=0.55,
            ),
        ]

        floor = CheckerboardMaterial(
            ambient_coefficient=1.0,
            diffuse_coefficient=0.8,
            square_size=1.2,
            white_color=Color(0.84, 0.84, 0.82),
            black_color=Color(0.24, 0.22, 0.20),
        )
        wall = SimpleMaterial(
            ambient_coefficient=0.18,
            diffuse_coefficient=0.72,
            diffuse_color=Color(0.63, 0.60, 0.56),
            specular_coefficient=0.08,
            specular_color=Color(1, 1, 1),
            specular_shininess=16,
        )
        wall_back = SimpleMaterialWithShadows(
            0.12,
            0.55,
            Color(0.48, 0.45, 0.42),
            0.1,
            Color(1, 1, 1),
            16,
        )

        self.add(Plane(point=Vector3D(0, 0, 0), normal=Vector3D(0, 0, 1)), floor)
        self.add(Plane(point=Vector3D(0, 12.0, 0), normal=Vector3D(0, -1, 0)), wall_back)
        self.add(Plane(point=Vector3D(0, 0, 0), normal=Vector3D(0, 1, 0)), wall)

        db = MerlBrdfDatabase()
        material_names = _pick_materials(db.available_materials(), 5)
        display_materials = [
            MerlBrdfMaterial(material_names[0], db, gain=14.0) if len(material_names) > 0 else None,
            MerlBrdfMaterial(material_names[1], db, gain=14.0) if len(material_names) > 1 else None,
            MerlBrdfMaterial(material_names[2], db, gain=14.0) if len(material_names) > 2 else None,
            MerlBrdfMaterial(material_names[3], db, gain=14.0) if len(material_names) > 3 else None,
            MerlBrdfMaterial(material_names[4], db, gain=14.0) if len(material_names) > 4 else None,
        ]
        display_materials = [material for material in display_materials if material is not None]

        group_centers = [
            Vector3D(-7.8, -0.6, 0.0),
            Vector3D(-3.9, 0.2, 0.0),
            Vector3D(0.0, -0.2, 0.0),
            Vector3D(3.9, 0.15, 0.0),
            Vector3D(7.8, -0.7, 0.0),
        ]
        group_scales = [0.92, 1.0, 1.18, 1.0, 0.9]
        group_yaws = [math.radians(-18), math.radians(8), math.radians(-6), math.radians(14), math.radians(-12)]

        for center, scale, yaw, material in zip(group_centers, group_scales, group_yaws, display_materials):
            _add_teapot_like_group(self, material, center, scale=scale, yaw=yaw)


if __name__ == "__main__":
    scene = Scene()
    print(f"Scene created. Objects: {len(scene.shapes)}")
    print(f"Available BRDFs: {len(MerlBrdfDatabase().available_materials())}")
