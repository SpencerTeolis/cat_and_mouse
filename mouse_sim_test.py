import numpy as np
import cv2
import time
from scene import Boundary
from mouse import Mouse, repel_vector, towards_cat
from util_funcs import *


points = np.load("calibration/calibration_points.npy")
# p = np.transpose(points)
# mask = p[0]
# mask[mask<100] = 0
# mask[mask>1000] = 0
# points = np.transpose(p[:,mask.astype(bool)])
point = np.asarray([300,400])

kwargs = {'calibration_points' : points, 'middle_point' : point}
boundary = Boundary(**kwargs)

dispW=1280 
dispH=720
disp_shape = (dispH,dispW,3)

img = np.zeros(disp_shape)
for x,y in boundary.lines:
    cv2.line(img,tuple(x),tuple(y),(0,0,255),2)

mouse_img = np.zeros_like(img)
mouse = Mouse(point, boundary, repel_vector, towards_cat)
mouse.draw_mouse(mouse_img)

def cat(event, x, y, flags, param):
    img, cat_pos = param
    if event == 0:
        #print(x,y)
        img[img[:,:,1].astype(bool)] = [0, 0, 0]
        cv2.circle(img, (x,y), 6, (0,255,0),-1)
        cat_pos[0] = x
        cat_pos[1] = y

cat_pos = point + 100
cv2.namedWindow("image")
cv2.setMouseCallback("image", cat, (img, cat_pos))

curr_time = time.time()
while True:
    #if time.time() - curr_time > .015:
    mouse.update_position(cat_pos)
    mouse_img = mouse.draw_mouse((dispH,dispW,3))
    cv2.imshow("image", img+mouse_img)
    curr_time = time.time()
    key = cv2.waitKey(300) & 0xFF 

    if key == ord("q"):
        break

cv2.destroyAllWindows()