import numpy as np
from scipy.spatial import ConvexHull

def normalize_points(arr):
    shape = arr.shape
    arr = arr.reshape(-1,2)
    arr = arr/np.linalg.norm(arr,axis=1).reshape(-1,1)
    return arr.reshape(shape)

def distance_from_line(lines, normals, point): 
    # vectors from start of line segment to given point
    vec1 = point - lines[:,0,:].reshape(-1,2)
    dot_p = np.sum(vec1*normals,axis=1)

    # n
    return(dot_p)

def distance(a,b):
    return np.linalg.norm(a-b)

def get_convex_hull_lines(points):
    hull = ConvexHull(points)
    idxs = np.stack((hull.vertices, np.roll(hull.vertices, -1)))

    return points[np.transpose(idxs)]