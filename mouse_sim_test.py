import numpy as np
import cv2
import time
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
mouse_img = np.zeros_like(bounds)
mouse = Mouse(point, boundary, repel_vector, towards_cat)
mouse.draw_mouse(mouse_img)

# Move 'cat' with cursor
def cat(event, x, y, flags, param):
    img, cat_pos = param
    if event == 0:
        cv2.circle(img, tuple(cat_pos), 6, (0,0,0),-1)
        cv2.circle(img, (x,y), 6, (0,255,0),-1)
        cat_pos[:] = x, y

cat_pos = point + 100
cv2.namedWindow("image")
cv2.setMouseCallback("image", cat, (bounds, cat_pos))

# Main loop
while True:
    mouse.update_position(cat_pos)
    mouse_img = mouse.draw_mouse(disp_shape)
    cv2.imshow("image", img+mouse_img)

    key = cv2.waitKey(300) & 0xFF 

    if key == ord("q"):
        break

cv2.destroyAllWindows()