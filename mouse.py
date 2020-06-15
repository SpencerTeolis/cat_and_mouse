import numpy as np
import cv2
import time
from scene import Boundary
from util_funcs import *

# Mouse class specifies how mouse should behave

class Mouse:

    def __init__(self, position, boundary: Boundary, evade_method, bait_method, momentum_scale=0.7, still_time=3):
        self.boundary = boundary
        self.position = position.astype(np.uint16)
        self.direction = np.zeros(2)
        self.dampening = momentum_scale
        self.evade_method = evade_method
        self.bait_method = bait_method
        self.evade = True
        self.last_move_time = time.time()
        self.still_time = still_time

    def update_position(self, cat_position):
        self.mode(cat_position)

        if self.evade:
            displacement = self.evade_method(self.boundary, self.position, cat_position)
            if np.sum(np.square(displacement)) > 6:
                self.last_move_time = time.time()
        else:
            displacement = self.bait_method(self.boundary, self.position, cat_position)
        
        self.direction = displacement + self.direction * self.dampening
        self.position = self.position + self.direction.astype(np.int16)

    def mode(self, cat_position):
        if time.time() - self.last_move_time < self.still_time or distance(self.position, cat_position) < 200:
            self.evade = True
        else:
            self.evade = False 

    def draw_mouse(self, disp_shape):
        # TODO fix for some reason passing (720,1280,3) as disp_shape gives 
        # "ValueError: maximum supported dimension for an ndarray is 32, found 720"
        disp_shape = (720,1280,3)
        img = np.zeros(disp_shape)
        cv2.circle(img,tuple(self.position),6,(0,0,255),-1)

        return img
        
def towards_cat(boundary: Boundary, mouse_pos, cat_pos):
    # TODO check if in bounds

    return (cat_pos-mouse_pos)/40

def towards_random_point(boundary: Boundary, mouse_pos, rand_point):
    pass

def repel_vector(boundary: Boundary, mouse_pos, cat_pos):
    cat_scale = 1
    boundary_scale = 0.75
    clip_value = 100

    dist = distance_from_line(boundary.lines, boundary.normals, mouse_pos).reshape(-1,1)
    dist = np.append(dist,distance(cat_pos, mouse_pos)/2)
    dist = dist/np.max(dist)
    
    mc_vec = normalize_points(mouse_pos-cat_pos)*cat_scale
    vecs = np.append(boundary.normals*boundary_scale,mc_vec.reshape(1,2),axis=0)

    mag = 1 / np.square(dist)
    mag = np.minimum(mag, clip_value)

    return np.sum(mag.reshape(-1,1) * vecs, axis=0)