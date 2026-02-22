# world is right-handed, z is up
import math
import random

from .ray import Ray

class Camera:
    def __init__(self, eye, look_at, up, fov, img_width, img_height):
        self.eye = eye
        # self.look_at = look_at
        # self.up = up
        # self.fov = fov
        # self.aspect_ratio = aspect_ratio
        self.img_width = img_width
        self.img_height = img_height

        aspect_ratio = img_height / img_width

        self.su = 2 * math.tan(math.radians(fov) / 2)
        self.sv = self.su * aspect_ratio

        self.w = (eye - look_at).normalize()
        up = up.normalize()
        #self.u = self.w.cross(up).normalize()
        self.u = up.cross(self.w).normalize()
        self.v = self.w.cross(self.u).normalize()

    def point_image2world(self, x, y):
        # from image coordinates to coordinates 
        # in the camera's view plane
        x_ndc = self.su * x / self.img_width - self.su / 2
        y_ndc = self.sv * y / self.img_height - self.sv / 2

        # from view plane to world coordinates
        return self.eye + self.u * x_ndc + self.v * y_ndc - self.w

    def ray(self, x, y):
        point_world = self.point_image2world(x, y)
        direction = (point_world - self.eye).normalize()
        return Ray(self.eye, direction)


class Camera_with_focal_depth:
    def __init__(self, eye, look_at, up, fov, img_width, img_height, radius, focal_dist):
        self.focal_dist = focal_dist
        self.radius = radius
        self.eye = eye
        # self.look_at = look_at
        # self.up = up
        # self.fov = fov
        # self.aspect_ratio = aspect_ratio
        self.img_width = img_width
        self.img_height = img_height

        aspect_ratio = img_height / img_width

        self.su = 2 * math.tan(math.radians(fov) / 2)
        self.sv = self.su * aspect_ratio

        self.w = (eye - look_at).normalize()
        up = up.normalize()
        #self.u = self.w.cross(up).normalize()
        self.u = up.cross(self.w).normalize()
        self.v = self.w.cross(self.u).normalize()

    def point_image2world(self, x, y):
        # from image coordinates to coordinates 
        # in the camera's view plane
        x_ndc = self.su * x / self.img_width - self.su / 2
        y_ndc = self.sv * y / self.img_height - self.sv / 2

        # from view plane to world coordinates
        return self.eye + self.u * x_ndc + self.v * y_ndc - self.w

    def ray(self, x, y):
        point_world = self.point_image2world(x, y)
        r = self.radius * math.sqrt(random.random())
        theta = 2 * math.pi * random.random()
        ue = r * math.cos(theta)
        ve = r * math.sin(theta)
        lens_point = self.eye + self.u * ue + self.v * ve
        p_f = (point_world - self.eye) * self.focal_dist + self.eye

        return Ray(lens_point, (p_f - lens_point).normalize())


        