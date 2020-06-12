from adafruit_servokit import ServoKit
from collections import deque
from scipy import interpolate
import cv2
import numpy as np
import time
import random
import os

def nothing(x):
    pass

cv2.namedWindow('TrackBars')
cv2.moveWindow('TrackBars', 0,0)

trackbar_names = ['RedLow','RedHigh','GreenLow','GreenHigh','BlueLow','BlueHigh']
init_vals = [183,255,100,255,191,255]
max_val = 255

for name, val in zip(trackbar_names, init_vals):
    cv2.createTrackbar(name, 'TrackBars',val,max_val,nothing)

kit = ServoKit(channels=16)
#3264x2464
dispW=1280 
dispH=720
flip=0

camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam=cv2.VideoCapture(camSet)

# Define threshold values for laser
ret, frame = cam.read()
print(frame.shape)
while True:
    ret, frame = cam.read()

    l_b = np.asarray([cv2.getTrackbarPos(name,'TrackBars') for name in trackbar_names[::2]])
    u_b = np.asarray([cv2.getTrackbarPos(name,'TrackBars') for name in trackbar_names[1::2]])

    laserMask = cv2.inRange(frame,l_b,u_b)
    cv2.imshow('laserMask', laserMask)
    cv2.moveWindow('laserMask',610,610)

    if cv2.waitKey(1)==ord('q'):
        break

cv2.destroyAllWindows()

# Move laser out of the way
kit.servo[0].angle = 0
kit.servo[1].angle = 0
time.sleep(2)

# Get other objects in the scene that have the same threshold values as the laser
blackMask = np.zeros((dispH,dispW))
for i in range(50):
    ret, frame = cam.read()
    time.sleep(0.01)
    blackMask = np.logical_or(blackMask, cv2.inRange(frame,l_b-30,u_b))

blackMask = blackMask.astype(np.uint8)*255
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
blackMask = cv2.dilate(blackMask,kernel,iterations=1) 

curr_time = time.time()

# Define grid of pan and tilt coordinates to calibrate on 
min_pan, max_pan, min_tilt, max_tilt, step =  20, 181, 120, 181, 5
pt_grid = np.transpose(np.mgrid[min_pan:max_pan:step,min_tilt:max_tilt:step]).reshape(-1,2)

# grid of xy coordinates to fill in during calibration 
xy_grid = np.zeros_like(pt_grid)

i = 0
max_i = pt_grid.shape[0]
flip = 1
while True:
    ret, frame = cam.read()
    if time.time() - curr_time > .25:

        if flip > 0:
            p,t = pt_grid[i%max_i]
            kit.servo[0].angle = p
            kit.servo[1].angle = t

        else:
            ret, frame = cam.read()
            laserMask = cv2.inRange(frame,l_b,u_b)
            laserMask[blackMask==255] = False
            cv2.imshow('laserMask', laserMask)

            M = cv2.moments(laserMask)
            cX, cY = 0, 0
            if M['m00'] != 0:
                print("not zero")
                cX = int(M['m10']/M['m00']) 
                cY = int(M['m01']/M['m00'])         

            print(np.sum(laserMask)/255)

            xy_grid[i%max_i] = [cX, cY]
            i += 1

        flip *= -1
        curr_time=time.time()

    if cv2.waitKey(1)==ord('q'):
        break

cv2.destroyAllWindows()

X,Y = np.mgrid[0:dispW+1,0:dispH+1]

points_mask = np.sum(xy_grid, axis=1).astype(bool)
points = xy_grid[points_mask,:]
pan_values = pt_grid[points_mask,0]
tilt_values = pt_grid[points_mask,1]

interp_p = interpolate.griddata(points, pan_values, (X, Y))
interp_t = interpolate.griddata(points, tilt_values, (X, Y))

os.makedirs("calibration",exist_ok=True)
np.save("calibration/pan_values.npy", pan_values)
np.save("calibration/tilt_values.npy", tilt_values)
np.save("calibration/calibration_points.npy", points)
np.save("calibration/interp_p.npy", interp_p)
np.save("calibration/interp_t.npy", interp_t)

img_overlay = np.zeros((dispH,dispW))
img_overlay[points[:,1], points[:,0]] = 255
img_overlay = cv2.dilate(img_overlay,kernel,iterations=1) 
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
cv2.namedWindow('nanoCam')
cv2.setMouseCallback('nanoCam', move_servos)
while True:
    ret, frame = cam.read()
    frame[img_overlay.astype(bool)] = [255,0,0]
    cv2.imshow('nanoCam', frame)
    
    if cv2.waitKey(1)==ord('q'):
        break


cam.release()
cv2.destroyAllWindows()