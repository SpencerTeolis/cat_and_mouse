# File to test if calibration worked by allowing user to control servos with mouse
# Laser point should lie approximately on the screen where the mouse is

import numpy as np
import cv2
import time
from adafruit_servokit import ServoKit

# set up servos
kit = ServoKit(channels=16)

# set up camera and image window
dispW=1280 
dispH=720
flip=0
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam=cv2.VideoCapture(camSet)
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

curr_time = [time.time()]
cv2.setMouseCallback('image', move_servos)
while True:
    ret, frame = cam.read()
    frame[img_overlay.astype(bool)] = [255,0,0]
    cv2.imshow('image', frame)
    
    if cv2.waitKey(1)==ord('q'):
        break


cam.release()
cv2.destroyAllWindows()