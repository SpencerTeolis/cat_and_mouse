import numpy as np
from scipy.spatial import ConvexHull
from util_funcs import normalize_points, magnitude

class Boundary:

    def __init__(self, **kwargs):
        self.lines = self.get_convex_hull_lines(kwargs["calibration_points"])
        self.interior_point = kwargs["middle_point"]

        self.edge_vecs = self.lines[:,1,:] - self.lines[:,0,:]
        self.interior_vecs = self.interior_point - self.lines[:,0,:].reshape(-1,2)
        self.normals = self.__get_unit_normals()
        self.areas = self.__get_triangle_areas()

    def __get_unit_normals(self):
        perpendicular = np.zeros_like(self.edge_vecs)
        perpendicular[:,0] = -self.edge_vecs[:,1]
        perpendicular[:,1] = self.edge_vecs[:,0]

        dot_p = np.sum(self.interior_vecs * perpendicular, axis=1)
        # Make unit length
        normals = normalize_points(perpendicular)
        # Change Normals to face towards point
        return normals * (dot_p / np.abs(dot_p)).reshape(-1,1) # shape n, 2

    def __get_triangle_areas(self):
        base = magnitude(self.edge_vecs, axis=1)
        height = np.sum(self.normals * self.interior_vecs, axis=1)

        return base * height / 2

    def get_convex_hull_lines(self, points):
        hull = ConvexHull(points)
        idxs = np.stack((hull.vertices, np.roll(hull.vertices, -1)))

        return points[np.transpose(idxs)]

    def in_bounds(self, point) -> bool:
        vec1 = point - self.lines[:,0,:].reshape(-1,2)
        dot_p = np.sum(vec1 * self.normals, axis=1)

        return (dot_p >= 0).all()

    def get_point_in_triangle(self, vec1, vec2, uniform=True):
        a1, a2 = np.random.uniform(0, 1, size=2)
        if uniform:
            while a2 > 1-a1:
                a1, a2 = np.random.uniform(0, 1, size=2)
        else:
            a2 *= (1-a1)

        return a1 * vec1 + a2 * vec2

    # If not uniform points tend towards the center 
    def get_point_in_bounds(self, uniform=True):
        probs = self.areas/np.sum(self.areas)
        triangle_idx = np.random.choice(len(self.areas), p=probs)
        vec1 = self.interior_vecs[triangle_idx]
        vec2 = self.edge_vecs[triangle_idx]

        return self.get_point_in_triangle(vec1, vec2, uniform) + self.lines[triangle_idx ,0]
