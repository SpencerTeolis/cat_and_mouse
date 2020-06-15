import numpy as np
from util_funcs import *

class Boundary:

    def __init__(self, **kwargs):
        self.lines = get_convex_hull_lines(kwargs["calibration_points"])
        self.normals = self.__get_unit_normals(self.lines, kwargs["middle_point"])

    def __get_unit_normals(self, lines, point):
        diff = lines[:,0,:] - lines[:,1,:]
        normals = np.zeros_like(diff)
        normals[:,0] = -diff[:,1]
        normals[:,1] = diff[:,0]

        vec1 = point - lines[:,0,:].reshape(-1,2)
        dot_p = np.sum(vec1*normals,axis=1)

        # Make unit length
        normals = normalize_points(normals)

        # Change Normals to face towards point
        normals = normals * (dot_p/np.abs(dot_p)).reshape(-1,1)

        # n, 2
        return normals
