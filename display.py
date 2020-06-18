import numpy as np
import cv2

def set_camera(dispW=1280 , dispH=720, flip=0, fps=30):
    camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width='+str(dispW)+', height='+str(dispH)+', format=NV12, framerate='+str(fps)+'/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
    return cv2.VideoCapture(camSet)

def display_cam(cam, window_name, img_function, wait=1, **kwargs):
    while True:
        ret, frame = cam.read()
        img = img_function(frame, **kwargs)
        cv2.imshow(window_name, img)
        
        if cv2.waitKey(wait)==ord('q'):
            break

def do_with_cam(cam, window_name, img_function, wait=1, **kwargs):
    while True:
        ret, frame = cam.read()
        img_function(frame, **kwargs)
        
        if cv2.waitKey(wait)==ord('q'):
            break

def display_img(img, window_name, img_function, wait=1, **kwargs):
    while True:
        img_function(img, **kwargs)
        cv2.imshow(window_name, img)
        
        if cv2.waitKey(wait)==ord('q'):
            break