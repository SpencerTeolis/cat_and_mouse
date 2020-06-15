import numpy as np
import cv2
import time
from adafruit_servokit import ServoKit
from scene import Boundary
from mouse import Mouse, repel_vector, towards_cat
from util_funcs import *

def nothing(x):
    pass

cv2.namedWindow('TrackBars')
cv2.moveWindow('TrackBars', 0,0)

trackbar_names = ['RedLow','RedHigh','GreenLow','GreenHigh','BlueLow','BlueHigh']
init_vals = [0,70,0,100,0,64]
max_val = 255

for name, val in zip(trackbar_names, init_vals):
    cv2.createTrackbar(name, 'TrackBars',val,max_val,nothing)

kit = ServoKit(channels=16)
#3264x2464
dispW=1280 
dispH=720
flip=0

camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam=cv2.VideoCapture(camSet)

# Define threshold values for tracking object
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

# Get other objects in the scene that have the same threshold values as the tracking object
blackMask = np.zeros((dispH,dispW))
for i in range(50):
    ret, frame = cam.read()
    time.sleep(0.01)
    blackMask = np.logical_or(blackMask, cv2.inRange(frame,l_b,u_b+10))

blackMask = blackMask.astype(np.uint8)*255
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
blackMask = cv2.dilate(blackMask,kernel,iterations=1) 


points = np.load("calibration/calibration_points.npy")
p = np.transpose(points)
mask = p[0]
mask[mask<100] = 0
mask[mask>1000] = 0
points = np.transpose(p[:,mask.astype(bool)])

interp_p = np.nan_to_num(np.load("calibration/interp_p.npy"))
interp_t = np.nan_to_num(np.load("calibration/interp_t.npy"))

kwargs = {'calibration_points' : points, 'middle_point' : point}
boundary = Boundary(**kwargs)

bounds = np.zeros((dispH,dispW))
for x,y in boundary.lines:
    cv2.line(bounds,tuple(x),tuple(y),255,2)

mouse = Mouse(point, boundary, repel_vector, towards_cat)
dispBounds = np.asarray([dispW,dispH])

def get_cat_pos(frame):
    momoMask = cv2.inRange(frame,l_b,u_b)
    momoMask[blackMask==255] = False

    M = cv2.moments(momoMask)
    cX, cY = 0, 0
    if M['m00'] != 0:
        cX = int(M['m10']/M['m00']) 
        cY = int(M['m01']/M['m00']) 

    # img = np.zeros((dispH,dispW,3))
    # img[momoMask.astype(bool),0] = 255
    # img[bounds.astype(bool),1] = 255
    # img[mouse.astype(bool),2] = 255

    # cv2.circle(img,(cX, cY),6,(255,255,0),-1)
    # cv2.imshow('momoMask', img)
    return cX, cY

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