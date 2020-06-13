import numpy as np
import cv2
import time
from adafruit_servokit import ServoKit
from scipy.spatial import ConvexHull

def normalize_points(arr):
    shape = arr.shape
    arr = arr.reshape(-1,2)
    arr = arr/np.linalg.norm(arr,axis=1).reshape(-1,1)
    return arr.reshape(shape)

def distance_from_line(lines, normals, point): 
    vec1 = point - lines[:,0,:].reshape(-1,2)
    dot_p = np.sum(vec1*normals,axis=1)

    # n
    return(dot_p)

def distance(a,b):
    return np.linalg.norm(a-b)

def get_unit_normals(lines, point):
    diff = lines[:,0,:] - lines[:,1,:]
    normals = np.zeros_like(diff)
    normals[:,0] = -diff[:,1]
    normals[:,1] = diff[:,0]

    vec1 = point - lines[:,0,:].reshape(-1,2)
    dot_p = np.sum(vec1*normals,axis=1)

    # Make unit length
    normals = normalize_points(normals)

    # Change Normals to face towards point
    normals = normals * (dot_p/np.abs(dot_p)).reshape(-1,1)

    # n, 2
    return normals

def get_repel_vector(lines, normals, cat, mouse):
    cat_scale = 2
    boundary_scale = 0.75
    clip_value = 100

    dist = distance_from_line(lines, normals, mouse).reshape(-1,1)
    dist = np.append(dist,distance(cat, mouse)/2)
    dist = dist/np.max(dist)
    
    mc_vec = normalize_points(mouse-cat)*cat_scale
    vecs = np.append(normals*boundary_scale,mc_vec.reshape(1,2),axis=0)

    mag = 1 / np.square(dist)
    mag = np.minimum(mag, clip_value)

    return np.sum(mag.reshape(-1,1) * vecs, axis=0)

def get_convex_hull_lines(points):
    hull = ConvexHull(points)
    idxs = np.stack((hull.vertices, np.roll(hull.vertices, -1)))

    return points[np.transpose(idxs)]

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

lines = get_convex_hull_lines(points)
point = np.asarray([300,400])
normals = get_unit_normals(lines,point)

bounds = np.zeros((dispH,dispW))
for x,y in lines:
    cv2.line(bounds,tuple(x),tuple(y),255,2)


mouse_pos = point.astype(np.uint16)
mouse_dir = np.zeros(2)
mouse = np.zeros_like(bounds)
dispBounds = np.asarray([dispW,dispH])
cv2.circle(mouse,tuple(mouse_pos),6,255,-1)

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

def move_mouse(lines, normals, cat_pos, cat_still_count, mouse_pos, mouse_dir):
    if cat_still_count >= 120 and cat_still_count <= 180:
        displacement = (cat_pos-mouse_pos)/40
    else:
        dampening = 0.7
        displacement = get_repel_vector(lines,normals,cat_pos,mouse_pos) + dampening*mouse_dir

    mouse_pos = mouse_pos + displacement
    mouse_pos = np.minimum(np.maximum(mouse_pos,0),dispBounds).astype(np.uint16)
    mouse = np.zeros_like(bounds)
    #cv2.circle(mouse,tuple(mouse_pos),6,255,-1)
    pan = interp_p[mouse_pos[0], mouse_pos[1]]
    tilt = interp_t[mouse_pos[0], mouse_pos[1]]
    if pan and tilt:
        kit.servo[0].angle=pan
        kit.servo[1].angle=tilt
    return mouse, mouse_pos, displacement

curr_time = time.time()
cat_still_count = 0
last_cat_pos = np.zeros(2,np.uint16)
cat_still_dist_const = 8

while True:
    ret, frame = cam.read()

    if time.time() - curr_time > 0.015:
        cat_pos = np.asarray(get_cat_pos(frame))
        mouse, mouse_pos, mouse_dir = move_mouse(lines,normals,cat_pos,cat_still_count,mouse_pos,mouse_dir)
        curr_time = time.time()

        if distance(last_cat_pos,cat_pos) < 10:
            cat_still_count += 1
        else:
            cat_still_count = 0
            last_cat_pos = np.copy(cat_pos)
        

    #cv2.imshow("image", img+mouse)
    key = cv2.waitKey(1) & 0xFF 

    if key == ord("q"):
        break

cv2.destroyAllWindows()