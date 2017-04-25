import glob
import pickle
import numpy as np
import cv2

# import time
import datetime

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

'''
Camera Calibration Methods
'''
def calc_calibration(cal_image_files, DEBUG_FLAG=False):

    nx = 9
    ny = 6

    if len(cal_image_files) > 0:
        test_image = cv2.imread(cal_image_files[0])
        img_size = (test_image.shape[0], test_image.shape[1])

    # Generte an array includes index of (x,y,z)
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    objpoints = []
    imgpoints = []

    if DEBUG_FLAG == True:
        start = datetime.datetime.now()
        print("start:{}".format(start))

    for idx, file_name in enumerate(cal_image_files):
        
        img = cv2.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # cv2.findChessboardCorners costs 0.2 ~ 0.5 second to execute ? 
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            if DEBUG_FLAG == True:
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                cv2.imshow(file_name, img)
                key = cv2.waitKey(0)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    dist_pickle["image_size"] = img_size
    pickle.dump( dist_pickle, open( "calibration_mtx_dist_pickle.p", "wb" ) )

    if DEBUG_FLAG == True:
        end = datetime.datetime.now()
        print('end:{}'.format(end))
        print('time elapse:{}'.format(end-start))
        cv2.destroyAllWindows()

    return dist_pickle


'''
Color Threshold Methods
'''
def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255), sigma=0.33, mode='manual'):
    # Calculate directional gradient
    # print('abs_sobel')
    # Default is using values of thresh
    lower = thresh[0]
    upper = thresh[1]
    if mode == 'auto':
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        # print('lower: ', lower)
        # print('upper: ', upper)
    
    if len(image.shape) == 2:
        target_image = image
    else:
        target_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    if orient == 'x':
        gradient = cv2.Sobel(target_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=sobel_kernel)
    if orient == 'y':
        gradient = cv2.Sobel(target_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=sobel_kernel)

    abs_gradient = np.absolute(gradient)
    scaled_gradient = np.uint8(255*abs_gradient/np.max(abs_gradient))
    # print("scaled_gradient", scaled_gradient)
    grad_binary = np.zeros_like(scaled_gradient)
    grad_binary[(scaled_gradient >= lower) & (scaled_gradient <= upper)] = 1
    # print("gard_bianry", grad_binary)
    return grad_binary

def mag_thresh(image, sobel_kernel=3, thresh=(0, 255), sigma=0.33, mode='manual'):
    # Calculate gradient magnitude
    # print('mag_thresh')
    lower = thresh[0]
    upper = thresh[1]
    if mode == 'auto':
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        # print('lower: ', lower)
        # print('upper: ', upper)
    
    if len(image.shape) == 2:
        target_image = image
    else:
        target_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    gx = cv2.Sobel(target_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=sobel_kernel)
    gy = cv2.Sobel(target_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=sobel_kernel)
    mag_gradient = np.sqrt(gx**2 + gy**2)
    # print('mag_gradient', mag_gradient)
    mag_gradient = np.uint8(mag_gradient/np.max(mag_gradient)*255)

    mag_binary = np.zeros_like(mag_gradient)
    mag_binary[(mag_gradient >= lower) & (mag_gradient <= upper)] = 1

    return mag_binary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    if len(image.shape) == 2:
        target_image = image
    else:
        target_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(target_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=sobel_kernel)
    gy = cv2.Sobel(target_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=sobel_kernel)
    atan_gardient = np.arctan2(np.absolute(gy), np.absolute(gx))
    dir_binary = np.zeros_like(atan_gardient)
    dir_binary[(atan_gardient > thresh[0]) & (atan_gardient < thresh[1])] = 1  
    return dir_binary

'''
Perspective transform (Bird view) Method
'''
def warper(image, four_point_src, four_point_dst):

    image_size = (image.shape[1], image.shape[0])

    transform_matrix = cv2.getPerspectiveTransform(four_point_src, four_point_dst)
    warped = cv2.warpPerspective(image, transform_matrix, image_size)

    return warped, transform_matrix


'''
Detect Line methods
'''
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    
    y_start = int(img_ref.shape[0]-(level+1)*height)
    y_end = int(img_ref.shape[0]-level*height)
    
    x_start = max(0, int(center-width/2))
    x_end = min(int(center+width/2), img_ref.shape[1])
    
    output[y_start:y_end, x_start:x_end] = 1
    
    return output

def find_window_centroids_window_search(binary_img, left_line, right_line):
    half = np.int(binary_img.shape[0]/2)

    # calculate histogram (vector) of the bottom half of s_binary
    # hostogram is a vector 
    histogram = np.sum(binary_img[half:, :], axis=0)
    out_image = np.dstack((binary_img, binary_img, binary_img))*255

    midpoint = int(histogram.shape[0]/2)

    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # number of sliding windows
    nwindows = 9

    window_height = np.int(binary_img.shape[0]/nwindows)
    # print("window_height:{}".format(window_height))
    # numpy.ndarray.nonzero(): Return the indices of the elements that are non-zero
    #                          the return result is a 2-D array
    #                          array[0] includes indices of row
    #                          array[1] includes indices of column
    # nonzero 返回的 array[0] 是 非零元素的 y index
    #               array[1] 是 非零元素的 x index
    # 所以 (array[0][3], array[1][3]) = 第 4 筆 非零元素的 index
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    # set the width of windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # 由 image 下到上移動 sliding window
        win_y_low = binary_img.shape[0] - (window+1)*window_height
        win_y_high = binary_img.shape[0] - window*window_height
        
        # 左半邊的 sliding window
        win_xleft_low = int(leftx_current - margin)
        win_xleft_high = int(leftx_current + margin)

        # 右半邊的 sliding window
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        # 畫左半邊的 sliding window
        cv2.rectangle(out_image, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 3)
        # 畫右半邊的 sliding window
        cv2.rectangle(out_image, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 3)
        
        '''
        nonzerox 跟 nonzeroy 存的是 binary image 裡非零元素的 x index 跟 y index
        如果 a = np.array([1,2,3,4,5,6,7,8,9])
        b = a<5, 則 
        b = array([ True,  True,  True,  True, False, False, False, False, False], dtype=bool)
        b.nonzero() = [0, 1, 2, 3]
        '''
        
        # 找出在左半邊 sliding window 裡所有非零元素的 index 
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        # 找出在右半邊 sliding window 裡所有非零元素的 index 
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # 將在左半邊 sliding window 裡所有非零元素的 index 加到 list 供之後算 polynomial 畫出左邊的線道
        left_lane_inds.append(good_left_inds)
        # 將在右半邊 sliding window 裡所有非零元素的 index 加到 list 供之後算 polynomial 畫出右邊的線道
        right_lane_inds.append(good_right_inds)
        
        # 取 左半邊跟右半邊計算 下一個 sliding window 的 X 軸位置，這裡是取平均值
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    #         print("{}: left_current:{}".format(window, leftx_current))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    #         print("{}: right_current:{}".format(window, rightx_current))

    # print('left_lane_inds[0]', left_lane_inds[0])

    # 因為 left_lane_inds, right_lane_inds 是一個 list 包含 nwindows 這麼多個 one dimension 的 numpy.ndarray
    # 所以將 letf_lane_inds 轉成 numpy.ndarray 並同時將這 nwindows 個 numpy.ndarray 以 axis = 0 
    # 串起來，所以這裡用 np.concatenate()
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # print("left_lane_inds[0]:{}".format(left_lane_inds[0]))

    # left_lane_inds, right_lane_inds 存的是 nonzerox 跟 nonzeroy 裡滿足每個 sliding window 範圍裡的 index
    # 而 nonzerox, nonzeroy 裡存的是 image array 裡非零元素的 index
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order poluynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_img.shape[0]-1, binary_img.shape[0] )
    # print('ploty', ploty.shape)
    # fit 2-degree polynomial curve equation
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return (left_fit, left_fitx, right_fit, right_fitx, ploty)



def find_window_centroids_conv(image, window_width, window_height, margin):
    window_centroids = []
    window = np.ones(window_width)
    
    midpoint_x = int(image.shape[1]/2)
    quarter_size = image.shape[0]/4
    offset = window_width / 2
    
    # np.convolve(a, v, mode='full'): 
    # Returns the discrete, linear convolution of two one-dimensional sequences.
    
    # First Layer for Left Window
    l_sum = np.sum(image[int(3*quarter_size):, :midpoint_x], axis=0)
    l_center_index = np.argmax(np.convolve(window, l_sum)) - offset
        
    # First Layer for Right Window
    r_sum = np.sum(image[int(3*quarter_size):, midpoint_x:], axis=0)
    r_center_index = np.argmax(np.convolve(window, r_sum)) - offset + midpoint_x

    window_centroids.append((l_center_index, r_center_index))
    
    
    for level in range(1, (int)(image.shape[0]/window_height)):
        y_start = int(image.shape[0] - (level+1)*window_height)
        y_end = int(image.shape[0] - level*window_height)

        # take histogram of full width of image
        image_layer = np.sum(image[y_start: y_end, :], axis=0)
        
        # take convolution with histogram
        conv_signal = np.convolve(window, image_layer)
        
        # Use window_width/2 as offset because convolution signal reference is at right side of window, 
        # not center of window
        
        # use past center as reference
        l_min_index = int(max(l_center_index + offset - margin, 0))
        l_max_index = int(min(l_center_index + offset + margin, image.shape[1]))
        l_center_index = np.argmax(conv_signal[l_min_index: l_max_index]) + l_min_index - offset
        
        r_min_index = int(max(r_center_index + offset - margin, 0))
        r_max_index = int(min(r_center_index + offset + margin, image.shape[1]))
        r_center_index = np.argmax(conv_signal[r_min_index: r_max_index]) + r_min_index - offset
        
        window_centroids.append((l_center_index, r_center_index))
    
    return window_centroids

def fit_lane_line(window_centroids, image_size):
    # window_centroids is a list of tuple of (l_center_index, r_center_index)
    # image_size is (image_width, image_height)
    
    image_height = image_size[1]
    # Get Centroid of Left Lane
    leftx = np.array([left for (left, right) in window_centroids])
    # Get Centroid of Right Lane
    rightx = np.array([right for (left, right) in window_centroids])

    leftx = leftx[::-1]
    rightx = rightx[::-1]
    ploty = np.linspace(0, image_height-1, leftx.shape[0])

    # Fit a second order polynomial to pixel positions in each lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return (left_fit, left_fitx, right_fit, right_fitx)

def draw_lane_field(image, warped, four_point_src, four_point_dst, x_fits):
    (image_height, image_width) = (image.shape[0], image.shape[1])
    (left_fitx, right_fitx) = x_fits
    ploty = np.linspace(0, image_height-1, left_fitx.shape[0])
    
    invM = cv2.getPerspectiveTransform(four_point_dst, four_point_src)

    fit_all_x = np.hstack((left_fitx, right_fitx[::-1]))
    fit_all_y = np.hstack((ploty, ploty[::-1]))
    all_vertices = np.reshape(np.column_stack((fit_all_x, fit_all_y)), (1, -1, 2))

    lane_img = np.dstack((warped, warped, warped))*0
    cv2.fillPoly(lane_img, np.int32(all_vertices), (0, 255, 0))
    lane_img = cv2.warpPerspective(lane_img, invM, (image_width, image_height))

    result = cv2.addWeighted(image, 1, lane_img, 0.3, 0)
    return invM, result

def detect_lane_curvature_v1(ploty, fit_coeff, fit_x):
    (left_fit_coeff, right_fit_coeff) = fit_coeff
    
    y_eval = np.max(ploty)
    
    # print("Left_curverad\t: {}\nRight_curverad\t: {}".format(left_curverad, right_curverad))

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    leftx = fit_x[0]
    rightx = fit_x[1]
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print("left curverad\t: {} m \nright_curverad\t: {} m".format(left_curverad, right_curverad))
    return (left_curverad, right_curverad)

def detect_lane_curvature_v2(ploty, fit_coeff, window_centroids):

    (left_fit_coeff, right_fit_coeff) = fit_coeff
    
    y_eval = np.max(ploty)
    
    # left_curverad = ((1 + (2*left_fit_coeff[0]*y_eval + left_fit_coeff[1])**2)**1.5) / np.absolute(2*left_fit_coeff[0])
    # right_curverad = ((1 + (2*right_fit_coeff[0]*y_eval + right_fit_coeff[1])**2)**1.5) / np.absolute(2*right_fit_coeff[0])

    print("Left_curverad\t: {}\nRight_curverad\t: {}".format(left_curverad, right_curverad))

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    leftx = np.array([left for (left, right) in window_centroids])
    rightx = np.array([right for (left, right) in window_centroids])
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print("left curverad\t: {} m \nright_curverad\t: {} m".format(left_curverad, right_curverad))
    return (left_curverad, right_curverad)


if __name__ == '__main__':

    cali_image_files = glob.glob('camera_cal/calibration*.jpg')

    calc_calibration(cali_image_files, DEBUG_FLAG=True)

