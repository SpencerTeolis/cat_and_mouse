import numpy as np

def normalize_points(arr):
    shape = arr.shape
    arr = arr.reshape(-1,2)
    arr = arr/np.linalg.norm(arr,axis=1).reshape(-1,1)
    return arr.reshape(shape)

def distance_from_line(vertices, normals, point): 
    # vectors from start of line segment to given point
    vec1 = point - vertices
    dot_p = np.sum(vec1*normals,axis=1)

    # n
    return(dot_p)

def magnitude(a, axis=0):
    return np.linalg.norm(a, axis=axis)

def distance(a, b):
    return np.linalg.norm(a-b)

def in_range_video(frames, lower_bound, upper_bound):
    g = np.greater_equal(frames, lower_bound)
    g = g.all(axis=-1)

    l = np.less_equal(frames, upper_bound)
    l = l.all(axis=-1)

    return np.logical_and(g,l).any(axis=0)

def mask_img(arr, mask, replace_value = None):
    dispW, dispH, channels = arr.shape
    if replace_value is None:
        replace_value = np.zeros(channels, dtype=np.uint8)

    return np.where(mask.reshape(dispW,dispH,1), arr, replace_value)