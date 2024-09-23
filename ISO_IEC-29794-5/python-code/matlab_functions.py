import cv2
import numpy as np
import math


def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype

def rgb2gray(rgb):
        return np.array(np.dot(rgb[..., :3], [0.299, 0.587, 0.114]),dtype='uint8')


def LBP_3x3(gray_array):
    size = gray_array.shape
    if len(size)==3:
        I = rgb2gray(gray_array)
    else:
        I = gray_array
    lbpI = np.zeros((size[0],size[1]))
    for i in range(1,size[0]-1):
        for j in range(1,size[1]-1):
            neighbor = [I[i-1,j-1], I[i-1,j], I[i-1,j+1], I[i,j+1], I[i+1,j+1], I[i+1,j], I[i+1,j-1], I[i,j-1]] > I[i,j] 
            pixel = 0
            for k in range(1,9):
                pixel = pixel + neighbor[k-1] * (1<<8-k)
            lbpI[i,j] = int(pixel)
    return lbpI


def imgaborfilt(image=None,wavelength=2,ort=0,spacial_freq_bw=2,spacial_aspect_ratio=0.5):
    orientation = -ort/180 * math.pi    # in radian, and seems to run in opposite direction
    sigma = 0.5 * wavelength * spacial_freq_bw         # 1 == SpatialFrequencyBandwidth
    gamma = spacial_aspect_ratio                         # SpatialAspectRatio
    shape = 1 + 2 * math.ceil(4 * sigma) # smaller cutoff is possible for speed
    shape = (60,60)
    gabor_filter_real = cv2.getGaborKernel(shape, sigma, orientation, wavelength, gamma, psi=0)
    gabor_filter_imag = cv2.getGaborKernel(shape, sigma, orientation, wavelength, gamma, psi=math.pi/2)

    gabor = cv2.filter2D(image, -1, gabor_filter_real) + 1j * cv2.filter2D(image, -1, gabor_filter_imag)
    mag = np.abs(gabor)
    phase = np.angle(gabor)
    return mag,phase


# import numpy as np
# import math
# from skimage.filters import gabor
# def imgaborfilt(image=None,wavelength=2.0,ort=None,spacial_freq_bw=2.0,spacial_aspect_ratio=0.5):
#     """TODO (does not quite) reproduce the behaviour of MATLAB imgaborfilt function using skimage."""
#     sigma = 0.5 * wavelength * spacial_freq_bw
#     filtered_image_re, filtered_image_im = gabor(
#         image, frequency=1 / wavelength, theta=-ort / 180 * math.pi,
#         sigma_x=sigma, sigma_y=sigma/spacial_aspect_ratio, n_stds=3,
#     )
    
#     full_image = filtered_image_re + 1j * filtered_image_im
#     mag = np.abs(full_image)
#     mag = filtered_image_re
#     phase = np.angle(full_image)
#     return mag, phase