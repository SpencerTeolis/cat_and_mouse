# File to test if calibration worked by allowing user to control servos with mouse.
# Laser point should lie approximately on the screen where the mouse is.

import numpy as np
import cv2
import time
import display
from adafruit_servokit import ServoKit

# set up servos
kit = ServoKit(channels=16)

# set up camera and image window
dispW, dispH = 1280, 720
cam = display.set_camera(dispW, dispH)
cv2.namedWindow("image")
cv2.moveWindow("image", 200,0)

# Load servo coordinates to image coordinates maps
points = np.load("calibration/calibration_points.npy")
interp_p = np.nan_to_num(np.load("calibration/interp_p.npy"))
interp_t = np.nan_to_num(np.load("calibration/interp_t.npy"))

# Create overlay showing points sampled during calibration
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

curr_time = [time.time()]
cv2.setMouseCallback('image', move_servos)
display.display_cam(cam, 'image', update, img_overlay=img_overlay)

cam.release()
cv2.destroyAllWindows()