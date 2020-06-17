import numpy as np
import cv2
import time
from adafruit_servokit import ServoKit
from scene import Boundary
from mouse import Mouse, repel_vector, towards_cat
from tracking import TrackObject, Threshold
from util_funcs import *

kit = ServoKit(channels=16)
dispW=1280 
dispH=720
flip=0

# set up camera and image window
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam=cv2.VideoCapture(camSet)
cv2.namedWindow("image")
cv2.moveWindow("image", 200,0)

# Load servo coordinates to image coordinates maps
points = np.load("calibration/calibration_points.npy")
interp_p = np.nan_to_num(np.load("calibration/interp_p.npy"))
interp_t = np.nan_to_num(np.load("calibration/interp_t.npy"))

# Set bounds
kwargs = {'calibration_points' : points, 'middle_point' : np.asarray([dispW//2,dispH//2],np.uint16)}
boundary = Boundary(**kwargs)
bounds = np.zeros((dispH,dispW))
for x,y in boundary.lines:
    cv2.line(bounds,tuple(x),tuple(y),255,2)

# Create tracking object
thresh = TrackObject(Threshold,init_vals = [0,0,0,70,100,64])

# Get values for thresholding
while True:
    ret, frame = cam.read()

    img = thresh.get_segmentation(frame)
    cv2.imshow('image', img)

    if cv2.waitKey(1)==ord('q'):
        break

cv2.destroyAllWindows()

# Get other objects in the scene that have the same threshold values as the tracking object
frames = []
for i in range(50):
    frames.append(cam.read()[1])
    time.sleep(0.01)

thresh.set_background_mask(frames)

mouse = Mouse(point, boundary, repel_vector, towards_cat)
dispBounds = np.asarray([dispW,dispH])

def get_cat_pos(frame):
    img = thresh.get_segmentation(frame) 

    return thresh.get_midpoint(img.any(axis=-1).astype(np.uint8))

def move_mouse(mouse, cat_pos):
    mouse.update_position(cat_pos)

    x, y = np.minimum(np.maximum(mouse.position,0),dispBounds).astype(np.uint16)
    pan = interp_p[x, y]
    tilt = interp_t[x, y]
    if pan and tilt:
        kit.servo[0].angle=pan
        kit.servo[1].angle=tilt


curr_time = time.time()
while True:
    ret, frame = cam.read()

    if time.time() - curr_time > 0.015:
        cat_pos = np.asarray(get_cat_pos(frame))
        move_mouse(mouse, cat_pos)
        curr_time = time.time()

    #cv2.imshow("image", img+mouse)
    key = cv2.waitKey(1) & 0xFF 

    if key == ord("q"):
        break

cv2.destroyAllWindows()