import numpy as np
import cv2
from util_funcs import *

def nothing(x): pass

class TrackObject:

    def __init__(self, seg_method, **seg_kwargs):
        self.seg = seg_method(**seg_kwargs)

    def get_segmentation(self, frame):
        return self.seg.get_segmentation(frame)

    def get_mask(self, frame):
        return self.seg.get_mask(frame)
    
    def get_midpoint(self, mask):
        M = cv2.moments(mask)
        cX, cY = 0, 0
        if M['m00'] != 0:
            cX = int(M['m10']/M['m00']) 
            cY = int(M['m01']/M['m00']) 

        return cX, cY

class Threshold:

    def __init__(self, 
                 window_name = 'TrackBars', 
                 trackbar_names = ['BlueLow','GreenLow','RedLow','BlueHigh','GreenHigh','RedHigh'],
                 init_vals = [0,0,0,255,255,255],
                 max_vals = 255):

        """
        Keyword arguments:
        window_name -- the window name that will be populated with trackbars
        trackbar_names -- the names of each track bar in left to right low to high order
        init_vals -- initial values for each trackbar in left to right low to high order
        max_vals -- max value of each trackbar int or list of ints in left to right low to high order
        """

        self.window_name = window_name
        self.trackbar_names = trackbar_names
        self.__create_trackbars(init_vals, max_vals)
        self.bg_mask = None

    def __create_trackbars(self, init_vals, max_vals):
        if isinstance(max_vals, int): 
            max_vals = [max_vals] * len(self.trackbar_names)

        cv2.namedWindow(self.window_name)
        for tn, iv, mv in zip(self.trackbar_names, init_vals, max_vals):
            cv2.createTrackbar(tn, self.window_name, iv, mv, nothing)

    def get_trackbar_values(self):
        bounds = np.asarray([cv2.getTrackbarPos(name, self.window_name) for name in self.trackbar_names])

        return bounds.reshape(2,-1)

    def set_background_mask(self, frames):
        lower_bound, upper_bound = self.get_trackbar_values()
        lower_bound = np.minimum(lower_bound, lower_bound-2)
        upper_bound = np.maximum(upper_bound, upper_bound+2)
        
        if len(frames.shape) == 4:
            frames = np.stack(frames) # nframes, dispH, dispW, dispChannels

        self.bg_mask = in_range_video(frames, lower_bound, upper_bound)

    def get_mask(self, frame):
        lower_bound, upper_bound = self.get_trackbar_values()
        fg = cv2.inRange(frame, lower_bound, upper_bound)

        if self.bg_mask is None:
            return fg

        return mask_img(np.expand_dims(fg, axis=2), np.invert(self.bg_mask))

    def get_segmentation(self, frame):
        fg_mask = self.get_mask(frame)

        return mask_img(frame, fg_mask)