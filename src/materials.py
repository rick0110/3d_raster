import math

from .base import Color, CastEpsilon, Material
from .ray import Ray
from .vector3d import Vector3D

class ColorMaterial(Material):
    def __init__(self,
                diffuse_color: Color,
    ):
        super().__init__()
        self.diffuse_color = diffuse_color

    def shade(self, hit_record, scene):
        return self.diffuse_color

class SimpleMaterial(Material):
    def __init__(self,
                ambient_coefficient: float,
                diffuse_coefficient: float,
                diffuse_color: Color,
                specular_coefficient: float,
                specular_color: Color,
                specular_shininess: float = 32
    ):
        super().__init__()
        self.ambient_coefficient = ambient_coefficient
        self.diffuse_coefficient = diffuse_coefficient
        self.diffuse_color = diffuse_color
        self.specular_coefficient = specular_coefficient
        self.specular_color = specular_color
        self.specular_shininess = specular_shininess

    def shade(self, hit_record, scene):
        shaded_color = Color(0, 0, 0)
        # Ambient component
        amb_color = scene.ambient_light * self.ambient_coefficient 
        for light in scene.lights:
            light_vector = light.position() - hit_record.point

            # Diffuse component
            light_dir = light_vector.normalize()
            diff_intensity = max(hit_record.normal.dot(light_dir), 0)
            diff_color = (self.diffuse_color @ light.color) * (self.diffuse_coefficient * diff_intensity)

            # Specular component
            view_dir = (scene.camera.eye - hit_record.point).normalize()
            reflect_dir = (hit_record.normal * 2 * hit_record.normal.dot(light_dir) - light_dir).normalize()
            spec_intensity = max(view_dir.dot(reflect_dir), 0) ** self.specular_shininess
            spec_color = (self.specular_color @ light.color) * self.specular_coefficient * spec_intensity

            # Accumulate color contributions
            shaded_color += (amb_color + diff_color + spec_color) * light.intensity

        return shaded_color

class SimpleMaterialWithShadows(SimpleMaterial):
    def __init__(self, ambient_coefficient: float, diffuse_coefficient: float, diffuse_color: Color, specular_coefficient: float, specular_color: Color, specular_shininess: float = 32):
        super().__init__(ambient_coefficient, diffuse_coefficient, diffuse_color, specular_coefficient, specular_color, specular_shininess)

    def shade(self, hit_record, scene):
        shaded_color = Color(0, 0, 0)
        # Ambient component
        amb_color = scene.ambient_light * self.ambient_coefficient 
        for light in scene.lights:
            light_vector = light.position() - hit_record.point

            # add ambient component once
            shaded_color += amb_color * light.intensity

            # Shadow check
            shadow_ray = Ray(hit_record.point + hit_record.normal * CastEpsilon, light_vector.normalize())
            shadow_hit = scene.hit(shadow_ray)
            if shadow_hit.hit and shadow_hit.t < light_vector.length():
                continue  # In shadow, skip this light

            # Diffuse component
            light_dir = light_vector.normalize()
            diff_intensity = max(hit_record.normal.dot(light_dir), 0)
            diff_color = (self.diffuse_color @ light.color) * (self.diffuse_coefficient * diff_intensity)

            # Specular component
            view_dir = (scene.camera.eye - hit_record.point).normalize()
            reflect_dir = (hit_record.normal * 2 * hit_record.normal.dot(light_dir) - light_dir).normalize()
            spec_intensity = max(view_dir.dot(reflect_dir), 0) ** self.specular_shininess
            spec_color = (self.specular_color @ light.color) * self.specular_coefficient * spec_intensity

            # Accumulate color contributions
            shaded_color += (diff_color + spec_color) * light.intensity

        return shaded_color

class CheckerboardMaterial(SimpleMaterial):
    def __init__(self, ambient_coefficient: float, diffuse_coefficient: float, square_size: float, white_color: Color = Color(1,1,1), black_color: Color = Color(0,0,0)):
        super().__init__(ambient_coefficient, diffuse_coefficient, Color(0, 0, 0), 0, Color(0,0,0), 32)
        self.square_size = square_size
        self.white_color = white_color
        self.black_color = black_color

    def shade(self, hit_record, scene):
        shaded_color = Color(0, 0, 0)
        # Ambient component
        amb_color = scene.ambient_light * self.ambient_coefficient 
        for light in scene.lights:
            light_vector = light.position() - hit_record.point

            # add ambient component once
            shaded_color += amb_color * light.intensity

            # Shadow check
            shadow_ray = Ray(hit_record.point + hit_record.normal * CastEpsilon, light_vector.normalize())
            shadow_hit = scene.hit(shadow_ray)
            if shadow_hit.hit and shadow_hit.t < light_vector.length():
                continue  # In shadow, skip this light

            # Diffuse component from checkerboard pattern
            u = hit_record.uv.x / self.square_size
            v = hit_record.uv.y / self.square_size

            diffuse_color = self.black_color  # black
            if (int(math.floor(u)) + int(math.floor(v))) % 2 == 0:
                diffuse_color = self.white_color  # white

            light_dir = light_vector.normalize()
            diff_intensity = max(hit_record.normal.dot(light_dir), 0)
            diff_color = (diffuse_color @ light.color) * (self.diffuse_coefficient * diff_intensity)

            # Accumulate color contributions
            shaded_color += diff_color * light.intensity

        return shaded_color

class ReflectiveMaterial(SimpleMaterialWithShadows):
    def __init__(self, ambient_coefficient: float, diffuse_coefficient: float,
                 diffuse_color: Color, specular_coefficient: float,
                 specular_color: Color, specular_shininess: float = 32,
                 reflection_coefficient: float = 0.5,
                 back_ground_color = None):
        super().__init__(ambient_coefficient, diffuse_coefficient, diffuse_color,
                         specular_coefficient, specular_color, specular_shininess)
        self.reflection_coefficient = reflection_coefficient
        self.back_ground_color = back_ground_color

    def shade(self, hit_record, scene):
        local_color = super().shade(hit_record, scene)

        if hit_record.ray.depth < scene.max_depth:
            d = hit_record.ray.direction
            n = hit_record.normal
            # if the normal direction is pointing to the same side as no ray direction, return background color
            if n.dot(d) > 0:
                if self.back_ground_color:
                    return self.back_ground_color
                return scene.background
            reflect_dir = (d - n * 2 * d.dot(n)).normalize()
            reflect_origin = hit_record.point + n * CastEpsilon
            reflect_ray = Ray(reflect_origin, reflect_dir, hit_record.ray.depth + 1)
            reflect_hit = scene.hit(reflect_ray)
            if reflect_hit.hit:
                reflected_color = reflect_hit.material.shade(reflect_hit, scene)
            else:
                reflected_color = scene.background
            # Blend local and reflected
            return local_color * (1 - self.reflection_coefficient) + reflected_color * self.reflection_coefficient

        return local_color


class TranslucidMaterial(SimpleMaterial):
    def __init__(self, ambient_coefficient: float, diffuse_coefficient: float, diffuse_color: Color, specular_coefficient: float, specular_color: Color, specular_shininess: float = 32, transmission_coefficient: float = 0.5, refraction_index: float = 1.5):
        super().__init__(ambient_coefficient, diffuse_coefficient, diffuse_color, specular_coefficient, specular_color, specular_shininess)
        self.transmission_coefficient = transmission_coefficient
        self.refraction_index = refraction_index

    def shade(self, hit_record, scene):
        # Ambient component
        shaded_color = scene.ambient_light * self.ambient_coefficient 
        origin = hit_record.ray.origin
        view_dir = (origin - hit_record.point).normalize()

        # we assume that outside the object is air with refraction index = 1.0
        # this is a simplification. A more complete implementation would track
        # whether we are inside or outside the object and use the appropriate refraction indices
        eta = 1.0 / self.refraction_index

        # We need to handle the case when we are inside the object
        c = hit_record.normal.dot(view_dir)
        n = hit_record.normal
        if c < 0:
            # we are inside the object
            # flip normal and adjust eta
            n = -n
            eta = 1.0 / eta
            # we also need to flip c so refraction calculations work correctly
            c = -c

        for light in scene.lights:
            light_vector = light.position() - hit_record.point
            # # Diffuse component
            light_dir = light_vector.normalize()
            diff_intensity = max(n.dot(light_dir), 0)
            shaded_color += (self.diffuse_color @ light.color) * (self.diffuse_coefficient * diff_intensity)*light.intensity

            # # Specular component
            reflect_dir = (n * 2 * n.dot(light_dir) - light_dir).normalize()
            spec_intensity = max(view_dir.dot(reflect_dir), 0) ** self.specular_shininess
            shaded_color += (self.specular_color @ light.color) * self.specular_coefficient * spec_intensity*light.intensity

        transmitted_color = Color(1, 0, 0)
        if hit_record.ray.depth < scene.max_depth:
            # transmission component
            k = 1 - eta**2 * (1 - c**2)
            if k >= 0: # if k < 0 total internal reflection occurs
                refract_dir =  (-view_dir * eta  + n * (eta * c - math.sqrt(k))).normalize()
                transmission_ray = Ray(hit_record.point, refract_dir, hit_record.ray.depth + 1)
                transmission_hit = scene.hit(transmission_ray)
                if transmission_hit.hit:
                    transmission_material = transmission_hit.material
                    transmitted_color = transmission_material.shade(transmission_hit, scene) * self.transmission_coefficient
                else:
                    transmitted_color = scene.background * self.transmission_coefficient
        else:
            transmitted_color = Color(0, 1, 0)
            
            # Accumulate color contributions
        shaded_color += transmitted_color

        return shaded_color