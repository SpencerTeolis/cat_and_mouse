import numpy as np
import cv2
import time
import display
import sys
import argparse
from adafruit_servokit import ServoKit
from scene import Boundary
from mouse import Mouse, repel_all, towards_cat
from tracking import TrackObject, Threshold
from util_funcs import *

parser = argparse.ArgumentParser(description="Test segmentation and tracking methods on input from console.")
parser.add_argument("--dispH", type=int, default=720, help="Display height (default: 720)")
parser.add_argument("--dispW", type=int, default=1280, help="Display width (default: 1280)")
parser.add_argument("--seg_method", type=str, default="threshold", choices=["threshold"], help="segmentation method to use")

seg_methods = {"threshold" : Threshold}

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# Set up servos
kit = ServoKit(channels=16)

# Set up camera and image window
dispW, dispH = 1280, 720
cam = display.set_camera(dispW, dispH)
cv2.namedWindow("image")
cv2.moveWindow("image", 200,0)

# Create tracking object for laser
thresh = TrackObject(seg_methods[opt.seg_method])

# Load servo coordinates to image coordinates maps
points = np.load("calibration/calibration_points.npy")
interp_p = np.nan_to_num(np.load("calibration/interp_p.npy"))
interp_t = np.nan_to_num(np.load("calibration/interp_t.npy"))

# Set bounds
point = np.asarray([dispW//2,dispH//2],np.int16)
kwargs = {'calibration_points' : points, 'middle_point' : point}
boundary = Boundary(**kwargs)
pts = boundary.vertices.reshape(-1,1,2)

# Get other objects in the scene that have the same threshold values as the tracking object
thresh.seg.set_background_mask_cam(cam, num_frames=20, tolerance=5)

mouse = Mouse(point, boundary, repel_all, towards_cat)

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