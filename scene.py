import numpy as np
from scipy.spatial import ConvexHull
from util_funcs import normalize_points, magnitude

class Boundary:

    def __init__(self, **kwargs):
        self.vertices = self.get_convex_hull_vertices(kwargs["calibration_points"])
        self.interior_point = kwargs["middle_point"]

        self.edge_vecs = np.roll(self.vertices, -1, axis=0) - self.vertices
        self.interior_vecs = self.interior_point - self.vertices
        self.edge_normals = self.__get_unit_normals()
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
        height = np.sum(self.edge_normals * self.interior_vecs, axis=1)

        return base * height / 2

    def get_convex_hull_vertices(self, points):
        return points[ConvexHull(points).vertices]

    def project_on_normals(self, point):
        vec1 = point - self.vertices.reshape(-1,2)
        return np.sum(vec1 * self.edge_normals, axis=1)

    def in_bounds(self, point) -> bool:
        return (self.project_on_normals(point) >= 0).all()

    def get_point_in_triangle(self, vec1, vec2):
        a1, a2 = np.random.uniform(0, 1, size=2)
        while a2 > 1-a1:
            a1, a2 = np.random.uniform(0, 1, size=2)

        return a1 * vec1 + a2 * vec2

    def get_point_in_bounds(self):
        probs = self.areas/np.sum(self.areas)
        triangle_idx = np.random.choice(len(self.areas), p=probs)

        vec1 = self.interior_vecs[triangle_idx]
        vec2 = self.edge_vecs[triangle_idx]

        return self.get_point_in_triangle(vec1, vec2) + self.vertices[triangle_idx]

    def draw_attributes(self, img):
        import cv2
        for i, v in enumerate(self.vertices):
            cv2.circle(img, tuple(v), 6, (255,0,0),-1)
            cv2.line(img, tuple(v), tuple(self.edge_vecs[i].astype(np.int16) + v), (200,0,200), 1)
            cv2.line(img, tuple(v), tuple(self.interior_vecs[i].astype(np.int16) + v), (200,0,200), 1)
            cv2.arrowedLine(img, tuple(v), tuple((self.edge_normals[i] * 50).astype(np.int16) + v), (0,255,255), 1)