"""MERL BRDF database support.

This module loads the measured binary BRDF files in material_data/BRDFDatabase
and evaluates them in the same half-angle / difference-angle parameterization
used by the reference MERL reader.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .base import CastEpsilon, Color, Material
from .ray import Ray
from .vector3d import Vector3D


BRDF_SAMPLING_RES_THETA_H = 90
BRDF_SAMPLING_RES_THETA_D = 90
BRDF_SAMPLING_RES_PHI_D = 360

RED_SCALE = 1.0 / 1500.0
GREEN_SCALE = 1.15 / 1500.0
BLUE_SCALE = 1.66 / 1500.0

_EXPECTED_SAMPLES = (
    BRDF_SAMPLING_RES_THETA_H
    * BRDF_SAMPLING_RES_THETA_D
    * BRDF_SAMPLING_RES_PHI_D
    // 2
)

_DEFAULT_BRDF_ROOT = (
    Path(__file__).resolve().parent.parent / "material_data" / "BRDFDatabase" / "brdfs"
)


def _cross_product(v1, v2):
    return (
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    )


def _normalize(vector):
    length = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _rotate_vector(vector, axis, angle):
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)

    out = [vector[0] * cos_angle, vector[1] * cos_angle, vector[2] * cos_angle]

    temp = axis[0] * vector[0] + axis[1] * vector[1] + axis[2] * vector[2]
    temp = temp * (1.0 - cos_angle)

    out[0] += axis[0] * temp
    out[1] += axis[1] * temp
    out[2] += axis[2] * temp

    cross = _cross_product(axis, vector)
    out[0] += cross[0] * sin_angle
    out[1] += cross[1] * sin_angle
    out[2] += cross[2] * sin_angle

    return tuple(out)


def _std_coords_to_half_diff_coords(theta_in, phi_in, theta_out, phi_out):
    in_vec_z = math.cos(theta_in)
    in_proj = math.sin(theta_in)
    in_vec = _normalize((in_proj * math.cos(phi_in), in_proj * math.sin(phi_in), in_vec_z))

    out_vec_z = math.cos(theta_out)
    out_proj = math.sin(theta_out)
    out_vec = _normalize((out_proj * math.cos(phi_out), out_proj * math.sin(phi_out), out_vec_z))

    half = _normalize(
        (
            (in_vec[0] + out_vec[0]) / 2.0,
            (in_vec[1] + out_vec[1]) / 2.0,
            (in_vec[2] + out_vec[2]) / 2.0,
        )
    )

    theta_half = math.acos(half[2])
    phi_half = math.atan2(half[1], half[0])

    bi_normal = (0.0, 1.0, 0.0)
    normal = (0.0, 0.0, 1.0)
    temp = _rotate_vector(in_vec, normal, -phi_half)
    diff = _rotate_vector(temp, bi_normal, -theta_half)

    theta_diff = math.acos(diff[2])
    phi_diff = math.atan2(diff[1], diff[0])
    return theta_half, phi_half, theta_diff, phi_diff


def _theta_half_index(theta_half):
    if theta_half <= 0.0:
        return 0
    theta_half_deg = (theta_half / (math.pi / 2.0)) * BRDF_SAMPLING_RES_THETA_H
    temp = theta_half_deg * BRDF_SAMPLING_RES_THETA_H
    ret_val = int(math.sqrt(temp))
    if ret_val < 0:
        ret_val = 0
    if ret_val >= BRDF_SAMPLING_RES_THETA_H:
        ret_val = BRDF_SAMPLING_RES_THETA_H - 1
    return ret_val


def _theta_diff_index(theta_diff):
    tmp = int(theta_diff / (math.pi * 0.5) * BRDF_SAMPLING_RES_THETA_D)
    if tmp < 0:
        return 0
    if tmp < BRDF_SAMPLING_RES_THETA_D - 1:
        return tmp
    return BRDF_SAMPLING_RES_THETA_D - 1


def _phi_diff_index(phi_diff):
    if phi_diff < 0.0:
        phi_diff += math.pi
    half_phi_res = BRDF_SAMPLING_RES_PHI_D // 2
    tmp = int(phi_diff / math.pi * half_phi_res)
    if tmp < 0:
        return 0
    if tmp < half_phi_res - 1:
        return tmp
    return half_phi_res - 1


def _lookup_brdf_val(brdf, theta_in, phi_in, theta_out, phi_out):
    theta_half, phi_half, theta_diff, phi_diff = _std_coords_to_half_diff_coords(
        theta_in, phi_in, theta_out, phi_out
    )
    _ = phi_half
    half_phi_res = BRDF_SAMPLING_RES_PHI_D // 2
    ind = (
        _phi_diff_index(phi_diff)
        + _theta_diff_index(theta_diff) * half_phi_res
        + _theta_half_index(theta_half)
        * half_phi_res
        * BRDF_SAMPLING_RES_THETA_D
    )

    n = _EXPECTED_SAMPLES
    red_val = brdf[ind] * RED_SCALE
    green_val = brdf[ind + n] * GREEN_SCALE
    blue_val = brdf[ind + 2 * n] * BLUE_SCALE
    return Color(red_val, green_val, blue_val)


def _world_to_local(direction, tangent, bitangent, normal):
    return Vector3D(
        direction.dot(tangent),
        direction.dot(bitangent),
        direction.dot(normal),
    )


def _orthonormal_basis(normal):
    helper = Vector3D(0, 0, 1)
    if abs(normal.z) > 0.999:
        helper = Vector3D(0, 1, 0)
    tangent = helper.cross(normal).normalize()
    bitangent = normal.cross(tangent).normalize()
    return tangent, bitangent, normal


class MerlBrdfDatabase:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else _DEFAULT_BRDF_ROOT
        self._cache = {}

    def available_materials(self):
        if not self.root.exists():
            return []
        return sorted(path.stem for path in self.root.glob("*.binary"))

    def material_path(self, name_or_path):
        path = Path(name_or_path)
        if path.is_file():
            return path
        if path.suffix != ".binary":
            path = path.with_suffix(".binary")
        if not path.is_absolute():
            path = self.root / path.name
        return path

    def load(self, name_or_path):
        path = self.material_path(name_or_path)
        key = str(path.resolve())
        if key not in self._cache:
            self._cache[key] = self._read(path)
        return self._cache[key]

    def _read(self, path):
        with path.open("rb") as brdf_file:
            dims = np.fromfile(brdf_file, dtype=np.int32, count=3)
            if dims.size != 3:
                raise ValueError(f"Could not read BRDF dimensions from {path}")
            n = int(dims[0] * dims[1] * dims[2])
            if n != _EXPECTED_SAMPLES:
                raise ValueError(f"Unexpected BRDF dimensions {tuple(int(v) for v in dims)} in {path}")
            brdf = np.fromfile(brdf_file, dtype=np.float64, count=3 * n)
            if brdf.size != 3 * n:
                raise ValueError(f"Could not read BRDF payload from {path}")
        return brdf


class MerlBrdfMaterial(Material):
    def __init__(self, brdf_name, database=None, gain=10.0, ambient_strength=0.02):
        super().__init__()
        self.brdf_name = brdf_name
        self.database = database if database is not None else MerlBrdfDatabase()
        self.gain = gain
        self.ambient_strength = ambient_strength

    def shade(self, hit_record, scene):
        brdf = self.database.load(self.brdf_name)
        shaded_color = scene.ambient_light * self.ambient_strength

        normal = hit_record.normal.normalize()
        view_dir = (scene.camera.eye - hit_record.point).normalize()
        if normal.dot(view_dir) <= 0:
            return shaded_color

        tangent, bitangent, normal = _orthonormal_basis(normal)
        local_view = _world_to_local(view_dir, tangent, bitangent, normal)
        theta_out = math.acos(max(min(local_view.z, 1.0), -1.0))
        phi_out = math.atan2(local_view.y, local_view.x)

        for light in scene.lights:
            light_vector = light.position() - hit_record.point
            light_distance = light_vector.length()
            if light_distance <= CastEpsilon:
                continue

            light_dir = light_vector / light_distance
            if normal.dot(light_dir) <= 0:
                continue

            shadow_ray = Ray(hit_record.point + normal * CastEpsilon, light_dir)
            shadow_hit = scene.hit(shadow_ray)
            if shadow_hit.hit and shadow_hit.t < light_distance:
                continue

            local_light = _world_to_local(light_dir, tangent, bitangent, normal)
            if local_light.z <= 0:
                continue

            theta_in = math.acos(max(min(local_light.z, 1.0), -1.0))
            phi_in = math.atan2(local_light.y, local_light.x)
            brdf_color = _lookup_brdf_val(brdf, theta_in, phi_in, theta_out, phi_out)
            shaded_color += (brdf_color @ light.color) * (light.intensity * local_light.z * self.gain)

        return shaded_color
