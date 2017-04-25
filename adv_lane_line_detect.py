# -*- coding:utf-8 -*-

import glob
import pickle
import os.path as path
import numpy as np
import cv2
# import time
import datetime

import utils

DEBUG_FLAG = False

# 1. Camera calibration

cal_image_files = glob.glob('camera_cal/calibration*.jpg')

if path.isfile("calibration_mtx_dist_pickle.p"):
    with open("calibration_mtx_dist_pickle.p", mode='rb') as fp:
        print("read from pickle")
        cali_dict = pickle.load(fp)
        mtx = cali_dict['mtx']
        dist = cali_dict['dist']
        img_size = cali_dict['image_size']

    print(mtx)
else:
    print("no pickle")
    
    cali_dict = utils.calc_calibration(cal_image_files)
    mtx = cali_dict['mtx']
    dist = cali_dict['dist']
    img_size = cali_dict['image_size']



# 2. Distortion correction
for image_file in cal_image_files:
    img = cv2.imread(image_file)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imshow(image_file, dst)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# 3. Color gradient threshold

# 4. Perspective transform

# 5. Detect lane lines
# 6. Determine the lane curvature



# test_image_files = glob.glob('test_images/*.jpg')