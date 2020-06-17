import numpy as np
import argparse
import cv2
import sys
import time
from tracking import TrackObject, Threshold


parser = argparse.ArgumentParser(description="Test segmentation and tracking methods on input from console.")
parser.add_argument("file_in", type=str, help="filename of the input image to process")
parser.add_argument("--dispH", type=int, default=0, help="Display height (default: given image height)")
parser.add_argument("--dispW", type=int, default=0, help="Display width (default: given image width)")
parser.add_argument("--seg_method", type=str, default="threshold", help="segmentation method to use")

seg_methods = {"threshold" : Threshold}

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

if opt.seg_method not in seg_methods:
    print("")
    print("invalid value for seg_method")
    print("supported methods:")
    for key in seg_methods:
        print(" - '{}'".format(key))
    sys.exit(0)

image = cv2.imread(opt.file_in, cv2.IMREAD_COLOR)
if opt.dispW | opt.dispH:
    image = cv2.resize(image, dsize=(opt.dispW, opt.dispH)) 



track = TrackObject(seg_methods[opt.seg_method])

while True:
    img = track.get_segmentation(np.copy(image))
    midpoint = track.get_midpoint(img.any(axis=-1).astype(np.uint8))
    cv2.circle(img, midpoint, 6, (0,255,0),-1)
    if cv2.waitKey(300)==ord('q'):
        break

    cv2.imshow('Mask', img)

cv2.destroyAllWindows()