import numpy as np
import cv2
import time
import display
from scene import Boundary
from mouse import Mouse, repel_vector, towards_cat
from util_funcs import *

dispH, dispW = 720, 1280
img = np.zeros((dispH, dispW, 3))

mouse_start = np.asarray([dispW//2, dispH//2])

# Set Boundary
boundary = Boundary(calibration_points = np.load("calibration/calibration_points.npy"), 
                    middle_point = mouse_start)
for p1, p2 in boundary.lines:
    cv2.line(img, tuple(p1), tuple(p2), (0,0,255), 2)

cat_pos = boundary.get_point_in_bounds().astype(np.int16)
cv2.circle(img, tuple(cat_pos), 6, (0,255,0),-1)

# Create mouse
mouse = Mouse(mouse_start, boundary, repel_vector, towards_cat)

# Move 'cat' with cursor
def cat(event, x, y, flags, param):
    img, cat_pos = param
    if event == 0:
        cv2.circle(img, tuple(cat_pos), 6, (0,0,0),-1)
        cv2.circle(img, (x,y), 6, (0,255,0),-1)
        cat_pos[:] = x, y

def update(img, mouse=None):
    cv2.circle(img, tuple(mouse.position), 6, (0,0,0),-1)
    mouse.update_position(cat_pos)
    cv2.circle(img, tuple(mouse.position), 6, (0,0,255),-1)

cv2.namedWindow("image")
cv2.setMouseCallback("image", cat, (img, cat_pos))
display.display_img(img, "image", update, mouse=mouse)