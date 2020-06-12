import numpy as np
import cv2
import time
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
    cat_scale = 3
    boundary_scale = 0.75
    clip_value = 200

    dist = distance_from_line(lines, normals, mouse).reshape(-1,1)
    dist = np.append(dist,distance(cat, mouse))
    dist = dist/np.max(dist)
    
    mc_vec = normalize_points(mouse-cat)*cat_scale#(mouse-cat)/np.linalg.norm(mouse-cat)*10
    vecs = np.append(normals*boundary_scale,mc_vec.reshape(1,2),axis=0)

    mag = 1 / np.square(dist)
    mag = np.minimum(mag, clip_value)

    return np.sum(mag.reshape(-1,1) * vecs, axis=0)

def get_convex_hull_lines(points):
    hull = ConvexHull(points)
    idxs = np.stack((hull.vertices, np.roll(hull.vertices, -1)))

    return points[np.transpose(idxs)]


points = np.load("calibration_points.npy")
p = np.transpose(points)
mask = p[0]
mask[mask<100] = 0
mask[mask>1000] = 0
points = np.transpose(p[:,mask.astype(bool)])

lines = get_convex_hull_lines(points)
#lines = np.asarray([[[20,20],[100,580]],[[20,20],[780,20]],[[100,580],[600,580]],[[780,20],[600,580]]])
point = np.asarray([300,400])
normals = get_unit_normals(lines,point)
print(distance_from_line(lines,normals,point))
print(normals)

print(get_repel_vector(lines,normals,point,point+50))

dispW=1280 
dispH=720

img = np.zeros((dispH,dispW,3))
for x,y in lines:
    cv2.line(img,tuple(x),tuple(y),(0,0,255),2)

mouse = np.zeros_like(img)
mouse_pos = point.astype(np.uint16)
cv2.circle(mouse,tuple(mouse_pos),6,(0,0,255),-1)

def circ_path(radius, cX, cY):
    a = np.linspace(0,2*np.pi,360)
    x = (cX + radius * np.cos(a)).astype(np.uint16)
    # cv2.imshow('momoMask', img)nt16)
    y = (cY + radius * np.sin(a)).astype(np.uint16)
    return np.transpose(np.stack((np.roll(x,-1) - x, np.roll(y,-1) - y)))

def cat(event, x, y, flags, param):
    img, cat_pos = param
    if event == 0:
        #print(x,y)
        img[img[:,:,1].astype(bool)] = [0, 0, 0]
        cv2.circle(img, (x,y), 6, (0,255,0),-1)
        cat_pos[0] = x
        cat_pos[1] = y

def move_mouse(lines,normals,cat_pos,mouse_pos, i, circ_dirs):
    displacement = get_repel_vector(lines,normals,cat_pos,mouse_pos)
    mouse_pos = mouse_pos + displacement #+ circ_dirs[i]/2

    mouse = np.zeros_like(img)
    cv2.circle(mouse,tuple(mouse_pos.astype(np.uint16)),6,(0,0,255),-1)
    return mouse, mouse_pos

circ_dirs = circ_path(10, dispW//2, dispH//2)
cat_pos = point + 100
cv2.namedWindow("image")
cv2.setMouseCallback("image", cat, (img, cat_pos))
i = 0
while True:
    mouse, mouse_pos = move_mouse(lines,normals,cat_pos,mouse_pos,i,circ_dirs)
    cv2.imshow("image", img+mouse)
    i+=1
    key = cv2.waitKey(1) & 0xFF 

    if key == ord("q"):
        break

cv2.destroyAllWindows()