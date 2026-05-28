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
from raytracer.shapes import Ball, Cilinder, Cube, ObjectTransform, Plane, PlaneUV, Translate
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


def _place(shape, offset, material, scene):
    scene.add(Translate(shape, offset), material)


def _add_building(scene, center, footprint=(1.0, 1.0), height=2.5, color=Color(0.55, 0.52, 0.48), roof_color=Color(0.35, 0.31, 0.30)):
    wall = SimpleMaterialWithShadows(0.08, 0.8, color, 0.12, Color(1, 1, 1), 24)
    roof = SimpleMaterialWithShadows(0.08, 0.7, roof_color, 0.1, Color(1, 1, 1), 20)
    body = ObjectTransform(Cube(size=1.0), scale3(footprint[0], footprint[1], height))
    _place(body, center + Vector3D(0, 0, height / 2.0), wall, scene)
    top = ObjectTransform(Cube(size=1.0), rot_z(math.radians(45)) @ rot_x(math.radians(30)) @ scale3(footprint[0] * 1.05, footprint[1] * 1.05, 0.45))
    _place(top, center + Vector3D(0, 0, height + 0.22), roof, scene)


def _add_tower(scene, center, radius=0.55, height=4.0, color=Color(0.42, 0.40, 0.37)):
    material = SimpleMaterialWithShadows(0.08, 0.82, color, 0.16, Color(1, 1, 1), 28)
    body = ObjectTransform(Cilinder(radius=radius, height=height), scale3(1.0, 1.0, 1.0))
    _place(body, center + Vector3D(0, 0, height / 2.0), material, scene)
    cap = Ball(center=Vector3D(0, 0, 0), radius=radius * 0.72)
    _place(cap, center + Vector3D(0, 0, height + radius * 0.55), material, scene)


def _add_tree(scene, center, scale=1.0):
    trunk = SimpleMaterialWithShadows(0.07, 0.85, Color(0.34, 0.22, 0.12), 0.08, Color(1, 1, 1), 12)
    foliage = SimpleMaterialWithShadows(0.08, 0.82, Color(0.15, 0.32, 0.14), 0.08, Color(1, 1, 1), 18)
    _place(ObjectTransform(Cilinder(radius=0.08 * scale, height=0.9 * scale), scale3(1, 1, 1)), center + Vector3D(0, 0, 0.45 * scale), trunk, scene)
    _place(Ball(center=Vector3D(0, 0, 0), radius=0.45 * scale), center + Vector3D(0, 0, 1.05 * scale), foliage, scene)
    _place(Ball(center=Vector3D(0, 0, 0), radius=0.28 * scale), center + Vector3D(0.18 * scale, 0.04 * scale, 1.42 * scale), foliage, scene)


def _add_cloud(scene, center, scale=1.0):
    cloud = SimpleMaterial(0.15, 0.65, Color(0.95, 0.95, 0.96), 0.0, Color(1, 1, 1), 4)
    for dx, dy, dz, radius in [
        (0.0, 0.0, 0.0, 0.7),
        (0.6, 0.08, 0.05, 0.5),
        (-0.5, 0.0, -0.02, 0.45),
        (0.15, -0.18, 0.08, 0.42),
    ]:
        _place(Ball(center=Vector3D(0, 0, 0), radius=radius * scale), center + Vector3D(dx * scale, dy * scale, dz * scale), cloud, scene)


def _add_distant_mountain(scene, center, scale=1.0):
    mountain = SimpleMaterialWithShadows(0.08, 0.8, Color(0.35, 0.36, 0.34), 0.05, Color(1, 1, 1), 12)
    _place(Ball(center=Vector3D(0, 0, 0), radius=2.1 * scale), center, mountain, scene)
    _place(Ball(center=Vector3D(0, 0, 0), radius=1.4 * scale), center + Vector3D(1.2 * scale, 0.0, 0.4 * scale), mountain, scene)


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
            eye=Vector3D(0.0, -18.0, 7.0),
            look_at=Vector3D(0.0, 0.8, 1.8),
            up=Vector3D(0, 0, 1),
            fov=40,
            img_width=2400,
            img_height=1200,
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
            0.28,
            0.48,
            Color(0.72, 0.79, 0.90),
            0.06,
            Color(1, 1, 1),
            16,
        )

        self.add(
            PlaneUV(
                point=Vector3D(0, 0, 0),
                normal=Vector3D(0, 0, 1),
                forward_direction=Vector3D(1, 0, 0),
            ),
            floor,
        )
        self.add(Plane(point=Vector3D(0, 12.0, 0), normal=Vector3D(0, -1, 0)), wall_back)

        # atmospheric background: mountains, clouds and a city line
        for center, scale in [
            (Vector3D(-13.5, 8.5, 0.9), 1.2),
            (Vector3D(-8.0, 9.4, 0.8), 1.0),
            (Vector3D(-1.0, 10.0, 0.7), 0.95),
            (Vector3D(7.0, 9.3, 0.85), 1.05),
            (Vector3D(13.5, 8.7, 0.9), 1.15),
        ]:
            _add_distant_mountain(self, center, scale=scale)

        for center, scale in [
            (Vector3D(-11.0, 6.0, 9.5), 1.0),
            (Vector3D(-4.0, 7.2, 10.2), 1.15),
            (Vector3D(4.0, 6.7, 9.8), 1.05),
            (Vector3D(11.2, 5.8, 9.2), 1.0),
        ]:
            _add_cloud(self, center, scale=scale)

        building_specs = [
            (Vector3D(-13.8, 6.0, 0.0), (1.0, 1.2), 4.1, Color(0.48, 0.44, 0.40), Color(0.30, 0.28, 0.26)),
            (Vector3D(-11.0, 5.6, 0.0), (1.2, 1.0), 5.1, Color(0.57, 0.50, 0.42), Color(0.34, 0.30, 0.28)),
            (Vector3D(-8.4, 6.2, 0.0), (0.9, 0.9), 3.2, Color(0.45, 0.46, 0.52), Color(0.26, 0.27, 0.30)),
            (Vector3D(-5.8, 5.7, 0.0), (1.4, 1.1), 6.0, Color(0.52, 0.49, 0.46), Color(0.24, 0.24, 0.24)),
            (Vector3D(-3.0, 6.4, 0.0), (0.85, 0.85), 4.2, Color(0.63, 0.55, 0.45), Color(0.35, 0.30, 0.24)),
            (Vector3D(-0.4, 5.8, 0.0), (1.25, 1.1), 5.2, Color(0.42, 0.45, 0.50), Color(0.25, 0.26, 0.30)),
            (Vector3D(2.6, 6.0, 0.0), (1.1, 0.9), 3.4, Color(0.58, 0.52, 0.43), Color(0.31, 0.28, 0.25)),
            (Vector3D(5.4, 5.7, 0.0), (1.45, 1.15), 6.4, Color(0.50, 0.51, 0.58), Color(0.28, 0.29, 0.34)),
            (Vector3D(8.5, 6.3, 0.0), (0.95, 0.95), 3.8, Color(0.60, 0.53, 0.46), Color(0.34, 0.29, 0.26)),
            (Vector3D(11.4, 5.8, 0.0), (1.3, 1.0), 4.8, Color(0.47, 0.46, 0.42), Color(0.28, 0.26, 0.24)),
            (Vector3D(14.2, 6.1, 0.0), (1.0, 1.1), 3.3, Color(0.55, 0.57, 0.52), Color(0.32, 0.33, 0.30)),
        ]
        for center, footprint, height, body_color, roof_color in building_specs:
            _add_building(self, center, footprint=footprint, height=height, color=body_color, roof_color=roof_color)

        tower_specs = [
            (Vector3D(-12.8, 5.2, 0.0), 0.42, 5.4, Color(0.38, 0.36, 0.33)),
            (Vector3D(-2.0, 5.7, 0.0), 0.36, 6.3, Color(0.46, 0.43, 0.38)),
            (Vector3D(9.6, 5.4, 0.0), 0.4, 5.8, Color(0.41, 0.39, 0.36)),
        ]
        for center, radius, height, color in tower_specs:
            _add_tower(self, center, radius=radius, height=height, color=color)

        for center in [
            Vector3D(-14.0, -2.4, 0.0),
            Vector3D(-10.5, -1.8, 0.0),
            Vector3D(-7.0, -2.1, 0.0),
            Vector3D(6.8, -1.9, 0.0),
            Vector3D(10.6, -2.3, 0.0),
        ]:
            _add_tree(self, center, scale=1.0)

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

        # foreground props to break the empty floor
        for center, scale in [
            (Vector3D(-6.5, -4.8, 0.0), 0.7),
            (Vector3D(-2.0, -5.4, 0.0), 0.75),
            (Vector3D(3.8, -4.9, 0.0), 0.72),
            (Vector3D(9.0, -5.5, 0.0), 0.68),
        ]:
            _add_tree(self, center, scale=scale)

        _add_cloud(self, Vector3D(-5.0, 1.8, 10.6), scale=0.8)
        _add_cloud(self, Vector3D(6.3, 2.3, 10.8), scale=0.7)


if __name__ == "__main__":
    scene = Scene()
    print(f"Scene created. Objects: {len(scene.shapes)}")
    print(f"Available BRDFs: {len(MerlBrdfDatabase().available_materials())}")
