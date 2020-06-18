# File to calibrate servos for a given setup.
#
# User specfies thresholding values to segment the laser point, then frames from the camera are taken at a lattice 
# of servo coordinates where the laser point is segmented and the corresponding x and y coordinates are found.
# 
# The resulting samples are then linearly interpolated between to create a pan tilt value for each image pixel.
# These maps, the points sampled, and the values at each point are saved in the calibration folder to use later.  

from adafruit_servokit import ServoKit
from collections import deque
from scipy import interpolate
from tracking import TrackObject, Threshold
from util_funcs import *
import display
import sys
import cv2
import numpy as np
import time
import random
import os

# Set up servos
kit = ServoKit(channels=16)

# Set up camera and image window
dispW, dispH = 1280, 720
cam = display.set_camera(dispW, dispH)
cv2.namedWindow("image")
cv2.moveWindow("image", 200,0)

# Create tracking object for laser
thresh = TrackObject(Threshold, init_vals = [191,191,191,255,255,255])

# Get values for thresholding
display.display_cam(cam, "image", thresh.get_segmentation)

# Move laser out of the way
kit.servo[0].angle = 0
kit.servo[1].angle = 0
cam.read()
time.sleep(3)

# Get other objects in the scene that have the same threshold values as the tracking object
frames = []
for i in range(20):
    frames.append(cam.read()[1])
    time.sleep(0.01)

thresh.seg.set_background_mask(frames, 20)
cv2.imshow('image', thresh.seg.bg_mask.astype(np.uint8)*255)

if cv2.waitKey(0)==ord('q'):
    sys.exit(0)

# Define grid of pan and tilt coordinates to sample 
min_pan, max_pan, min_tilt, max_tilt, step =  20, 181, 120, 181, 5
pt_grid = np.transpose(np.mgrid[min_pan:max_pan:step,min_tilt:max_tilt:step]).reshape(-1,2)

# grid of xy coordinates to fill in during calibration 
xy_grid = np.zeros_like(pt_grid)

# Calibration loop
def calibrate(frame, xy_grid, pt_grid, state, max_i):
    img = thresh.get_segmentation(frame)
    i, last_time = state
    if time.time() - last_time > .25:
        if i%2 == 0:
            # Hack: the segmented frame and the corresponding servo position is one off so add 1
            p,t = pt_grid[(i//2)+1 % max_i]
            kit.servo[0].angle = p
            kit.servo[1].angle = t
        else:
            xy_grid[(i//2) % max_i] = thresh.get_midpoint(img.any(axis=-1).astype(np.uint8))

        state[:] = i+1, time.time()
    return img
    
state = [0, time.time()]
display.display_cam(cam, "image", calibrate, xy_grid=xy_grid, pt_grid=pt_grid, state=state, max_i=pt_grid.shape[0])

X,Y = np.mgrid[0:dispW+1,0:dispH+1]

# Get list of points in the image where a pan and tilt value was recorded
points_mask = xy_grid.any(axis=1)
points = xy_grid[points_mask,:]
pan_values = pt_grid[points_mask,0]
tilt_values = pt_grid[points_mask,1]

# Linearly interpolate between sampled points to create an image sized map of pan and tilt values
interp_p = interpolate.griddata(points, pan_values, (X, Y))
interp_t = interpolate.griddata(points, tilt_values, (X, Y))

# Save calibration values for defining servo range of motion and to use in cat_and_mouse.py
os.makedirs("calibration",exist_ok=True)
np.save("calibration/pan_values.npy", pan_values)
np.save("calibration/tilt_values.npy", tilt_values)
np.save("calibration/calibration_points.npy", points)
np.save("calibration/interp_p.npy", interp_p)
np.save("calibration/interp_t.npy", interp_t)

img_overlay = np.zeros((dispH,dispW))
img_overlay[points[:,1], points[:,0]] = 255
img_overlay = cv2.dilate(img_overlay, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1) 

def move_servos(event,x,y,flags,param):
    if time.time() - curr_time[0] >= .015:
        pan = interp_p[x,y]
        tilt = interp_t[x,y]
        print(pan,tilt)
        if event == cv2.EVENT_MOUSEMOVE and pan and tilt:
            kit.servo[0].angle=pan
            kit.servo[1].angle=tilt
            print(pan,tilt)
        
        curr_time[0] = time.time()

def update(frame, img_overlay):
    frame[img_overlay.astype(bool)] = [255,0,0]
    return frame

# Test if calibration worked
curr_time = [time.time()]
cv2.setMouseCallback('image', move_servos)
display.display_cam(cam, 'image', update, img_overlay=img_overlay)

cam.release()
cv2.destroyAllWindows()