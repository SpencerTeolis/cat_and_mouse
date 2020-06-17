import numpy as np
import cv2
import time
import display
from scene import Boundary
from mouse import Mouse, repel_vector, towards_cat
from util_funcs import *

disp_shape = (720,1280,3)
point = np.asarray([300,400])

# Set Boundary
boundary = Boundary(calibration_points = np.load("calibration/calibration_points.npy"), middle_point = point)
bounds = np.zeros(disp_shape)
for x,y in boundary.lines:
    cv2.line(bounds,tuple(x),tuple(y),(0,0,255),2)

# Create mouse
mouse = Mouse(point, boundary, repel_vector, towards_cat)
img = np.copy(bounds)

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

cat_pos = point + 100
cv2.namedWindow("image")
cv2.setMouseCallback("image", cat, (img, cat_pos))

display.display_img(img, "image", update, mouse=mouse)