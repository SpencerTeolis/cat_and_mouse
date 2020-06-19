import numpy as np
import cv2
import time
from scene import Boundary
from util_funcs import *

# Mouse class specifies how mouse should behave

class Mouse:

    def __init__(self, position, boundary: Boundary, evade_method, bait_method, still_time=3):
        self.boundary = boundary
        self.position = position.astype(np.int16)
        self.direction = np.zeros(2)
        self.evade_method = evade_method
        self.bait_method = bait_method
        self.evade = True
        self.last_move_time = time.time()
        self.still_time = still_time
        self.last_cat_pos = np.zeros(2, dtype=np.int16)

    def update_position(self, cat_position):
        self.mode(cat_position)

        args = [self.boundary, self.position, self.direction, cat_position]
        if self.evade:
            new_pos = self.evade_method(*args)
        else:
            new_pos = self.bait_method(*args)
        
        self.direction = new_pos - self.position
        self.position = new_pos.astype(np.int16)

    def mode(self, cat_position):
        if distance(self.last_cat_pos, cat_position) > 6:
            self.last_cat_pos = np.copy(cat_position)
            self.last_move_time = time.time()

        if (time.time() - self.last_move_time < self.still_time) or (distance(self.position, cat_position) < 100):
            self.evade = True
        else:
            self.evade = False 

        
def towards_cat(boundary: Boundary, mouse_pos,  mouse_dir, cat_pos):
    new_pos = mouse_pos + (cat_pos-mouse_pos)/40

    return new_pos if boundary.in_bounds(new_pos) else mouse_pos

def repel_all(boundary: Boundary, mouse_pos, mouse_dir, cat_pos):
    cat_scale = 1.5
    boundary_scale = 1
    clip_value = 100
    dampening = 0.5

    # Distance from mouse position to each boundary
    dist = distance_from_line(boundary.vertices, boundary.edge_normals, mouse_pos).reshape(-1,1)
    # Append mouse to cat distance to array of mouse to boundary distances 
    dist = np.append(dist,distance(cat_pos, mouse_pos)/2)
    # Scale distances so that the distance from the middle to an edge is about 1
    dist = dist/np.mean(magnitude(boundary.interior_vecs, axis=1))
    
    # Get direction vectors forces should be applied on
    mc_vec = normalize_points(mouse_pos-cat_pos)*cat_scale
    vecs = np.append(boundary.edge_normals * boundary_scale, mc_vec.reshape(1,2), axis=0)

    # Get magnitude of force/displacement proportional to 1/dist^2 with max of clip_value
    mag = np.minimum(1 / np.square(dist), clip_value)
    repel_vector = np.sum(mag.reshape(-1,1) * vecs, axis=0)
    new_pos = mouse_pos + repel_vector + (mouse_dir * dampening)

    return new_pos if boundary.in_bounds(new_pos) else mouse_pos