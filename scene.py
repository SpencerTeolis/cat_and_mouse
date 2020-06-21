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
        vec1 = point - self.vertices
        return np.sum(vec1 * self.edge_normals, axis=1)

    def in_bounds(self, point) -> bool:
        return (self.project_on_normals(point) >= 0).all()

    def get_point_in_triangle(self, idx, n=1):
        adj_idx = (idx+1) % len(self.interior_vecs)

        # Get point in quadrilateral defined by linear combination of vec1 and vec2
        a1, a2 = np.random.uniform(0, 1, size=(2,n))
        vec1 = -self.interior_vecs[idx]
        vec2 = -self.interior_vecs[adj_idx]
        p = a1.reshape(-1,1) * vec1 + a2.reshape(-1,1) * vec2 + self.interior_point

        # Check if point is in triangle - dotproduct is negative 
        p_vec = p - self.vertices[idx]
        dot_p = np.sum(p_vec * self.edge_normals[idx], axis=1).reshape(-1,1)
        
        # Get 180 degree rotation of points about center of quadrilateral
        v1, v2 = self.vertices[idx], self.vertices[adj_idx]
        mid = (v1 + (v2 - v1)/2).astype(np.int16)
        rotation = mid - (p - mid)

        # if point in triagle return it else return point rotated 180 degrees which is also in the triangle
        return np.where(dot_p<0, rotation, p)

    def get_point_in_bounds(self, n=1):
        probs = self.areas/np.sum(self.areas)
        triangle_idx = np.random.choice(len(self.areas), size=n, p=probs)

        return np.squeeze(self.get_point_in_triangle(triangle_idx, n))

    def draw_attributes(self, img, vertices=True, boundary=True, interior=True, normals=True):
        import cv2
        for i, v in enumerate(self.vertices):
            if vertices:
                cv2.circle(img, tuple(v), 6, (255,0,0),-1)
            if boundary:
                cv2.line(img, tuple(v), tuple(self.edge_vecs[i].astype(np.int16) + v), (200,0,200), 1)
            if interior:
                cv2.line(img, tuple(v), tuple(self.interior_vecs[i].astype(np.int16) + v), (200,0,200), 1)
            if normals:
                cv2.arrowedLine(img, tuple(v), tuple((self.edge_normals[i] * 50).astype(np.int16) + v), (0,255,255), 1)