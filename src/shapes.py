from src.vector3d import Vector3D
from .base import Shape, HitRecord, CastEpsilon
import numpy as np
from .ray import Ray

class Ball(Shape):
    def __init__(self, center, radius):
        super().__init__("ball")
        self.center = center
        self.radius = radius

    def hit(self, ray):
        # Ray-sphere intersection
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return HitRecord(False, float('inf'), None, None)
        else:
            hit, point, normal = False, None, None
            t = (-b - discriminant**0.5) / (2.0 * a)
            if t > CastEpsilon:
                hit = True
                point = ray.point_at_parameter(t)
                normal = (point - self.center).normalize()
            else:
                t = (-b + discriminant**0.5) / (2.0 * a)
                if t > CastEpsilon:
                    hit = True
                    point = ray.point_at_parameter(t)
                    normal = (point - self.center).normalize()

            return HitRecord(hit, t, point, normal)

class Plane(Shape):
    def __init__(self, point, normal):
        super().__init__("plane")
        self.point = point
        self.normal = normal.normalize()

    def hit(self, ray):
        denom = self.normal.dot(ray.direction)
        if abs(denom) > 1e-6:
            t = (self.point - ray.origin).dot(self.normal) / denom
            if t >= CastEpsilon:
                point = ray.point_at_parameter(t)
                return HitRecord(True, t, point, self.normal)
        return HitRecord(False, float('inf'), None, None)

class PlaneUV(Shape):
    def __init__(self, point, normal, forward_direction):
        super().__init__("plane")
        self.point = point
        self.normal = normal.normalize()
        self.forward_direction = forward_direction.normalize()
        # compute right direction
        self.right_direction = self.normal.cross(self.forward_direction).normalize()

    def hit(self, ray):
        denom = self.normal.dot(ray.direction)
        if abs(denom) > 1e-6:
            t = (self.point - ray.origin).dot(self.normal) / denom
            if t >= CastEpsilon:
                point = ray.point_at_parameter(t)
                # Calculate UV coordinates
                vec = point - self.point
                u = vec.dot(self.right_direction)
                v = vec.dot(self.forward_direction)
                uv = Vector3D(u, v, 0)
                return HitRecord(True, t, point, self.normal, uv=uv)
        return HitRecord(False, float('inf'), None, None)

class ImplicitFunction(Shape):
    def __init__(self, function):
        super().__init__("implicit_function")
        self.func = function

    def in_out(self, point):
        return self.func(point) <= 0

    def bissect(self, t_1, t_2, ray, depth_search):
        for _ in range(depth_search):
            point_1 = ray.point_at_parameter(t_1)
            mid_t = (t_1 + t_2) / 2
            mid_point = ray.point_at_parameter(mid_t)
            if abs(self.func(mid_point)) < CastEpsilon:
                break
            elif self.func(mid_point) * self.func(point_1) < 0:
                t_2 = mid_t
            else:
                t_1 = mid_t
        return (t_1 + t_2) / 2

class Cube(Shape):
    def __init__(self, size):
        super().__init__("cube")
        self.size = size

    def time_in_out(self, ray):
        x_min = -self.size / 2
        x_max = self.size / 2
        y_min = -self.size / 2
        y_max = self.size / 2
        z_min = -self.size / 2
        z_max = self.size / 2

        t_min_x = float('inf') if abs(ray.direction.x) < CastEpsilon else (x_min - ray.origin.x) / ray.direction.x
        t_max_x = float('inf') if abs(ray.direction.x) < CastEpsilon else (x_max - ray.origin.x) / ray.direction.x
        t_min_y = float('inf') if abs(ray.direction.y) < CastEpsilon else (y_min - ray.origin.y) / ray.direction.y
        t_max_y = float('inf') if abs(ray.direction.y) < CastEpsilon else (y_max - ray.origin.y) / ray.direction.y
        t_min_z = float('inf') if abs(ray.direction.z) < CastEpsilon else (z_min - ray.origin.z) / ray.direction.z
        t_max_z = float('inf') if abs(ray.direction.z) < CastEpsilon else (z_max - ray.origin.z) / ray.direction.z

        t_min = max(min(t_min_x, t_max_x), min(t_min_y, t_max_y), min(t_min_z, t_max_z))
        t_max = min(max(t_min_x, t_max_x), max(t_min_y, t_max_y), max(t_min_z, t_max_z))

        return t_min, t_max

    def hit(self, ray):
        x_min = -self.size / 2
        x_max = self.size / 2
        y_min = -self.size / 2
        y_max = self.size / 2
        z_min = -self.size / 2
        z_max = self.size / 2

        t_min_x = float('inf') if abs(ray.direction.x) < CastEpsilon else (x_min - ray.origin.x) / ray.direction.x
        t_max_x = float('inf') if abs(ray.direction.x) < CastEpsilon else (x_max - ray.origin.x) / ray.direction.x
        t_min_y = float('inf') if abs(ray.direction.y) < CastEpsilon else (y_min - ray.origin.y) / ray.direction.y
        t_max_y = float('inf') if abs(ray.direction.y) < CastEpsilon else (y_max - ray.origin.y) / ray.direction.y
        t_min_z = float('inf') if abs(ray.direction.z) < CastEpsilon else (z_min - ray.origin.z) / ray.direction.z
        t_max_z = float('inf') if abs(ray.direction.z) < CastEpsilon else (z_max - ray.origin.z) / ray.direction.z

        t_min = max(min(t_min_x, t_max_x), min(t_min_y, t_max_y), min(t_min_z, t_max_z))
        t_max = min(max(t_min_x, t_max_x), max(t_min_y, t_max_y), max(t_min_z, t_max_z))

        t = -1
        if t_min <= t_max and t_min > CastEpsilon:
            t = t_min
        elif t_min <= t_max and t_max >= CastEpsilon:
            t = t_max
        if t_min <= t_max and t > CastEpsilon:
            point = ray.point_at_parameter(t)
            if t < float('inf'):
                # Determine the normal based on which face was hit
                if abs(point.x - x_min) < 1e-6:
                    normal = Vector3D(-1, 0, 0)
                elif abs(point.x - x_max) < 1e-6:
                    normal = Vector3D(1, 0, 0)
                elif abs(point.y - y_min) < 1e-6:
                    normal = Vector3D(0, -1, 0)
                elif abs(point.y - y_max) < 1e-6:
                    normal = Vector3D(0, 1, 0)
                elif abs(point.z - z_min) < 1e-6:
                    normal = Vector3D(0, 0, -1)
                else:
                    normal = Vector3D(0, 0, 1)
                return HitRecord(hit=True, t=t, point=point, normal=normal)

        return HitRecord(hit=False, t=float('inf'), point=None, normal=None)

class Cilinder(Shape):
    def __init__(self, radius, height):
        super().__init__("cilinder")
        self.radius = radius
        self.height = height

    def hit(self, ray):
        # Ray-cylinder intersection (infinite cylinder along z-axis)
        a = ray.direction.x**2 + ray.direction.y**2
        b = 2 * (ray.origin.x * ray.direction.x + ray.origin.y * ray.direction.y)
        c = ray.origin.x**2 + ray.origin.y**2 - self.radius**2
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return HitRecord(False, float('inf'), None, None)
        
        if abs(a) < CastEpsilon:
            return HitRecord(False, float('inf'), None, None)
        
        t0 = (-b - discriminant**0.5) / (2*a)
        t1 = (-b + discriminant**0.5) / (2*a)
        __t = float('inf')
        for t in [t0, t1]:
            if t < __t:
                __t = t
        if __t > CastEpsilon and __t < float('inf'):
            point = ray.point_at_parameter(__t)
            if -self.height/2 <= point.z <= self.height/2:
                normal = Vector3D(point.x, point.y, 0).normalize()
                return HitRecord(True, __t, point, normal)
        
        # verify if the ray hits the caps of the cylinder
        if discriminant >= 0:
            init = min(t0, t1)
            end = max(t0, t1)
            p_0 = ray.point_at_parameter(init)
            p_1 = ray.point_at_parameter(end)
            z_0 = min(p_0.z, p_1.z)
            z_1 = max(p_0.z, p_1.z)
            

            t = float('inf')
            if z_0 <= -self.height/2 and z_1 >= -self.height/2:
                __t = float('inf') if abs(ray.direction.z) < CastEpsilon else (-self.height/2 - ray.origin.z) / ray.direction.z

                if __t > CastEpsilon and __t < t:
                    t = __t
                    normal = Vector3D(0, 0, -1)
                    point = ray.point_at_parameter(t)
            
            elif z_0 <= self.height/2 and z_1 >= self.height/2:
                __t = float('inf') if abs(ray.direction.z) < CastEpsilon else (self.height/2 - ray.origin.z) / ray.direction.z
                if __t > CastEpsilon and __t < t:
                    t = __t
                    normal = Vector3D(0, 0, 1)
                    point = ray.point_at_parameter(t)
            
            if t < float('inf'):
                return HitRecord(True, t, point, normal)
        return HitRecord(False, float('inf'), None, None)


class Translate(Shape):
    def __init__(self, shape: Shape, offset: Vector3D):
        super().__init__(f"translate_{shape.type}")
        self.shape = shape
        self.offset = offset

    def hit(self, ray):
        moved_origin = ray.origin - self.offset
        moved_ray = Ray(moved_origin, ray.direction)
        hit_rec = self.shape.hit(moved_ray)
        if hit_rec.hit:
            point = hit_rec.point + self.offset
            normal = hit_rec.normal
            return HitRecord(True, hit_rec.t, point, normal, uv=getattr(hit_rec, 'uv', None))
        return HitRecord(False, float('inf'), None, None)


class ObjectTransform(Shape):
    def __init__(self, shape: Shape, matrix):
        super().__init__(f"object_transform_{shape.type}")
        self.shape = shape
        self.transform_func = np.array(matrix)
        self.inverse_transform_func = np.linalg.inv(self.transform_func)

    def hit(self, ray):
        direction = np.array(ray.direction.as_list())
        origin = np.array(ray.origin.as_list())
        direction_inv = self.inverse_transform_func @ direction
        origin_inv = self.inverse_transform_func @ origin
        norm_dir_inv = np.linalg.norm(direction_inv)
        direction_inv = direction_inv / norm_dir_inv
        ray_in_obj_space = Ray(Vector3D(*origin_inv), Vector3D(*direction_inv))
        hit_rec = self.shape.hit(ray_in_obj_space)
        
        if hit_rec.hit:
            point_obj_space = np.array(hit_rec.point.as_list())
            normal_obj_space = np.array(hit_rec.normal.as_list())

            point_world_space = self.transform_func @ point_obj_space
            normal_world_space = self.inverse_transform_func.T @ normal_obj_space
            point = Vector3D(*point_world_space)
            normal = Vector3D(*normal_world_space).normalize()
            # ensure the normal faces against the incoming world-space ray direction
            if normal.dot(ray.direction) > 0:
                normal = -normal

            return HitRecord(True, hit_rec.t / norm_dir_inv, point, normal, uv=hit_rec.uv)
        
        return HitRecord(False, float('inf'), None, None)

class Paraboloid:
    """Local paraboloid shape (z = k*(x^2 + y^2))."""
    def __init__(self, k=0.5):
        self.k = k
        self.type = "paraboloid"

    def hit(self, ray):
        from src.base import HitRecord, CastEpsilon
        ox, oy, oz = ray.origin.x, ray.origin.y, ray.origin.z
        dx, dy, dz = ray.direction.x, ray.direction.y, ray.direction.z

        A = -self.k * (dx * dx + dy * dy)
        B = dz - 2 * self.k * (ox * dx + oy * dy)
        C = oz - self.k * (ox * ox + oy * oy)

        if abs(A) < CastEpsilon:
            if abs(B) < CastEpsilon:
                return HitRecord(False, float('inf'), None, None)
            t = -C / B
            if t > CastEpsilon:
                point = ray.point_at_parameter(t)
                normal = Vector3D(-2 * self.k * point.x, -2 * self.k * point.y, -1).normalize()
                return HitRecord(True, t, point, normal)
            return HitRecord(False, float('inf'), None, None)

        disc = B * B - 4 * A * C
        if disc < 0:
            return HitRecord(False, float('inf'), None, None)
        sqrt_d = disc ** 0.5
        t0 = (-B - sqrt_d) / (2 * A)
        t1 = (-B + sqrt_d) / (2 * A)
        candidates = [t for t in (t0, t1) if t > CastEpsilon]
        if not candidates:
            return HitRecord(False, float('inf'), None, None)
        t = min(candidates)
        point = ray.point_at_parameter(t)
        normal = Vector3D(-2 * self.k * point.x, -2 * self.k * point.y, -1).normalize()
        return HitRecord(True, t, point, normal)

def mitchel_function(point):
    x, y, z = point.as_list()
    return 4*(x**4 + (y**2+z**2)**2 + 17*x**2*(y**2+z**2)) - 20*(x**2+y**2+z**2) + 17

class Mitchel_func(ImplicitFunction):
    def __init__(self):
        super().__init__(mitchel_function)
    
    def grad(self, point):
        x, y, z = point.as_list()
        return (16*x**3 + 136*x*(y**2 + z**2) - 40*x,
                16*y*(y**2 + z**2) + 136*x**2*y - 40*y,
                16*z*(y**2 + z**2) + 136*x**2*z - 40*z)

class Mitchel(Mitchel_func):
    def __init__(self, n_splits_search=50, depth_bissect_search=30):
        super().__init__()
        self.type = "mitchel"
        self.n_splits_search = n_splits_search
        self.depth_bissect_search = depth_bissect_search
        self.bounding_box = Cube(4)

    def hit(self, ray):
        hit_rec = self.bounding_box.hit(ray)
        if not hit_rec.hit:
            return HitRecord(False, float('inf'), None, None)
        
        t_min, t_max = self.bounding_box.time_in_out(ray)
        
        p_min = ray.point_at_parameter(t_min)
        p_max = ray.point_at_parameter(t_max)

        interval = np.linspace(t_min, t_max, self.n_splits_search)

        for i, t_i in enumerate(interval[:-1]): 
            t_i_1 = interval[i + 1]
            p_ti = ray.point_at_parameter(t_i)
            p_ti_1 = ray.point_at_parameter(t_i_1)
            value_pti = self.func(p_ti)
            value_pti_1 = self.func(p_ti_1)
            if value_pti * value_pti_1 < 0:
                t_min = t_i
                t_max = t_i_1
                t_c = self.bissect(t_min, t_max, ray, self.depth_bissect_search)
                point = ray.point_at_parameter(t_c)
                normal = Vector3D(*self.grad(point)).normalize()
                if abs(self.func(point)) < CastEpsilon:
                    return HitRecord(True, t_c, point, normal)
        
        return HitRecord(False, float('inf'), None, None)


# Heart implicit function (module-level so multiprocessing can pickle it)
def heart_function(point):
    x, y, z = point.as_list()
    A = x**2 + (9.0/4.0) * y**2 + z**2 - 1
    return A**3 - x**2 * z**3 - (9.0/80.0) * y**2 * z**3


class Heart_func(ImplicitFunction):
    def __init__(self):
        super().__init__(heart_function)

    def grad(self, point):
        x, y, z = point.as_list()
        A = x**2 + (9.0/4.0) * y**2 + z**2 - 1
        # df/dx = 3*A^2 * 2x - 2x*z^3 = 2x*(3*A^2 - z^3)
        dx = 2 * x * (3 * (A ** 2) - z ** 3)
        # df/dy = 3*A^2 * (9/2*y) - 2*(9/80)*y*z^3 = y*((27/2)*A^2 - (9/40)*z^3)
        dy = y * ((27.0 / 2.0) * (A ** 2) - (9.0 / 40.0) * (z ** 3))
        # df/dz = 3*A^2 * 2z - 3*z^2*(x^2 + (9/80)*y^2) = 6*z*A^2 - 3*z^2*(x^2 + (9/80)*y^2)
        dz = 6 * z * (A ** 2) - 3 * (z ** 2) * (x ** 2 + (9.0 / 80.0) * (y ** 2))
        return (dx, dy, dz)


class Heart(Heart_func):
    def __init__(self, n_splits_search=50, depth_bissect_search=30):
        super().__init__()
        self.type = "heart"
        self.n_splits_search = n_splits_search
        self.depth_bissect_search = depth_bissect_search
        # bounding box large enough to contain the heart shape
        self.bounding_box = Cube(3)

    def hit(self, ray):
        hit_rec = self.bounding_box.hit(ray)
        if not hit_rec.hit:
            return HitRecord(False, float('inf'), None, None)
        
        t_min, t_max = self.bounding_box.time_in_out(ray)
        
        p_min = ray.point_at_parameter(t_min)
        p_max = ray.point_at_parameter(t_max)

        interval = np.linspace(t_min, t_max, self.n_splits_search)

        for i, t_i in enumerate(interval[:-1]): 
            t_i_1 = interval[i + 1]
            p_ti = ray.point_at_parameter(t_i)
            p_ti_1 = ray.point_at_parameter(t_i_1)
            value_pti = self.func(p_ti)
            value_pti_1 = self.func(p_ti_1)
            if value_pti * value_pti_1 < 0:
                t_min = t_i
                t_max = t_i_1
                t_c = self.bissect(t_min, t_max, ray, self.depth_bissect_search)
                point = ray.point_at_parameter(t_c)
                normal = Vector3D(*self.grad(point)).normalize()
                if abs(self.func(point)) < CastEpsilon:
                    return HitRecord(True, t_c, point, normal)
        
        return HitRecord(False, float('inf'), None, None)


        