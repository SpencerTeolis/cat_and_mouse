import numpy as np
import cv2
import time
import display
from adafruit_servokit import ServoKit
from scene import Boundary
from mouse import Mouse, repel_all, towards_cat
from tracking import TrackObject, Threshold
from util_funcs import *

# Set up servos
kit = ServoKit(channels=16)

# Set up camera and image window
dispW, dispH = 1280, 720
cam = display.set_camera(dispW, dispH)
cv2.namedWindow("image")
cv2.moveWindow("image", 200,0)

# Create tracking object for laser
thresh = TrackObject(Threshold,init_vals = [0,0,0,70,100,64])

# Get values for thresholding
display.display_cam(cam, "image", thresh.get_mask)

# Load servo coordinates to image coordinates maps
points = np.load("calibration/calibration_points.npy")
interp_p = np.nan_to_num(np.load("calibration/interp_p.npy"))
interp_t = np.nan_to_num(np.load("calibration/interp_t.npy"))

# Set bounds
point = np.asarray([dispW//2,dispH//2],np.int16)
p = np.transpose(points)
mask = p[0]
mask[mask<100] = 0
mask[mask>1000] = 0
points = np.transpose(p[:,mask.astype(bool)])
kwargs = {'calibration_points' : points, 'middle_point' : point}
boundary = Boundary(**kwargs)
pts = boundary.vertices.reshape(-1,1,2)

# Get other objects in the scene that have the same threshold values as the tracking object
frames = []
for i in range(20):
    frames.append(cam.read()[1])
    time.sleep(0.01)

thresh.seg.set_background_mask(frames, 5)
cv2.imshow('image', thresh.seg.bg_mask.astype(np.uint8)*255)

if cv2.waitKey(0)==ord('q'):
    sys.exit(0)

mouse = Mouse(point, boundary, repel_all, towards_cat)
dispBounds = np.asarray([dispW,dispH])

def update(frame, mouse, state):
    mask = thresh.get_mask(frame)
    cat_pos = thresh.get_midpoint(mask)

    if time.time() - state[0] > 0.015:
        mouse.update_position(cat_pos)
        x, y = mouse.position
        pan = interp_p[x, y]
        tilt = interp_t[x, y]
        if pan and tilt:
            kit.servo[0].angle=pan
            kit.servo[1].angle=tilt
        
        state[0] = time.time()

    cv2.polylines(frame,[pts],True,(0,0,255))
    cv2.circle(frame, tuple(mouse.position), 6, (0,0,255),-1)
    cv2.circle(frame, tuple(cat_pos), 6, (255,255,0),-1)
    return frame

state = [time.time()]
display.display_cam(cam, "image", update, mouse=mouse, state=state)

cam.release()
cv2.destroyAllWindows()