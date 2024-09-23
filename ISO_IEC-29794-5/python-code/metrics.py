from matlab_functions import im2double, LBP_3x3, imgaborfilt
import numpy as np
import PIL.Image
from math import ceil
import scipy.ndimage 
# from scipy.stats import entropy
import cv2


def sharpnessMetric(gray_array):
    G = im2double(gray_array)
    Gy,Gx=np.gradient(G)
    S=np.sqrt(Gx*Gx+Gy*Gy)
    sharpness=np.sum(S)/(Gx.shape[0]*Gx.shape[1])
    return sharpness

def getGlobalContrastFactor(gray_array):
    
    GCF = 0.0

    resolutions = np.array([1, 2, 4, 8, 16, 25, 50, 100, 200])

    LC = np.zeros(resolutions.shape)
    W = gray_array.shape[1]
    H = gray_array.shape[0]

    im=PIL.Image.fromarray(gray_array)
    rIm = PIL.Image.fromarray(gray_array)
    c = np.zeros(resolutions.shape)
    w = np.zeros(resolutions.shape)
    for i in range(1,len(resolutions)+1):
        # attempt at resizing as in the paper
        if i>1:
            rIm_w = max((ceil(float(1/(2**(i-1)))*gray_array.shape[1]),1))
            rIm_H = max((ceil(float(1/(2**(i-1)))*gray_array.shape[1]),1))
            # rIm = im.resize((rIm_w,rIm_H), resample=PIL.Image.BILINEAR)
            rIm = im.resize((rIm_w,rIm_H), resample=PIL.Image.BILINEAR)

        
        W = rIm.size[1]
        H = rIm.size[0]
        l = (np.array(rIm)/255) * 2.2
        
        # compute perceptual luminance L
        rL = 100 * np.sqrt(l)
        # compute local contrast for each pixel
        lc = 0.0
        for x in range(0,H):
            for y in range(0,W):
                if (x == 0) and (x == H-1):
                    if (y == 0) and (y == W-1):
                        lc = lc + 0
                    elif (y == 0):
                        lc = lc + abs(rL[x,y] - rL[x,y+1])
                    elif (y == W-1):
                        lc = lc + abs(rL[x, y] - rL[x,y-1])
                    else:
                        lc = lc + (abs(rL[x, y] - rL[x,y-1]) + abs(rL[x, y] - rL[x,y+1]))/2
                    
                elif (x == 0):
                    if (y == 0) and (y == W-1):
                        lc = lc + abs(rL[x, y] - rL[x+1,y])
                    elif (y == 0):
                        lc = lc + ( abs(rL[x, y] - rL[x,y+1]) + abs(rL[x, y] - rL[x+1, y]) )/2
                    elif (y == W-1):
                        lc = lc + ( abs(rL[x, y] - rL[x, y-1]) + abs(rL[x, y] - rL[x+1, y]) )/2
                    else:
                        lc = lc + ( abs(rL[x, y] - rL[x, y-1]) + abs(rL[x, y] - rL[x, y+1]) + abs(rL[x, y] - rL[x+1, y]) )/3
                    
                elif (x == H-1):
                    if (y == 0) and (y == W-1):
                        lc = lc + abs(rL[x, y] - rL[x-1, y])
                    elif (y == 0):
                        lc = lc + ( abs(rL[x, y] - rL[x, y+1]) + abs(rL[x, y] - rL[x-1, y]) )/2
                    elif (y == W-1):
                        lc = lc + ( abs(rL[x, y] - rL[x, y-1]) + abs(rL[x, y] - rL[x-1, y]) )/2
                    else:
                        lc = lc + ( abs(rL[x, y] - rL[x, y-1]) + abs(rL[x, y] - rL[x, y+1]) + abs(rL[x, y] - rL[x-1, y]) )/3
                else: # x > 1 and x < H:
                    if (y == 0) and (y == W-1):
                        lc = lc + ( abs(rL[x, y] - rL[x+1, y]) + abs(rL[x, y] - rL[x-1, y]) )/2
                    elif (y == 0):
                        lc = lc + ( abs(rL[x, y] - rL[x, y+1]) + abs(rL[x, y] - rL[x+1, y]) + abs(rL[x, y] - rL[x-1, y]) )/3
                    elif (y == W-1):
                        lc = lc + ( abs(rL[x, y] - rL[x, y-1]) + abs(rL[x, y] - rL[x+1, y]) + abs(rL[x, y] - rL[x-1, y]) )/3
                    else:
                        lc = lc + ( abs(rL[x, y] - rL[x, y-1]) + abs(rL[x, y] - rL[x, y+1]) + abs(rL[x, y] - rL[x-1, y]) + abs(rL[x, y] - rL[x+1, y]) )/4

        
        # compute average local contrast c
        c[i-1] = lc/(W*H)
        w[i-1]  = (-0.406385*(i/9)+0.334573)*(i/9)+ 0.0877526
        
        # compute global contrast factor
        LC[i-1]  = c[i-1] *w[i-1] 
        GCF = GCF + LC[i-1] 
    return GCF
        

def blurMetric(gray_array):


    I = gray_array.astype(float)
    y,x = gray_array.shape

    Hv = np.ones((1,9))/9
    Hh = np.transpose(Hv)

    B_Ver = scipy.ndimage.correlate(I,Hv, mode='constant')
    #blur the input image in vertical direction
    B_Hor = scipy.ndimage.correlate(I,Hh, mode='constant')
    #blur the input image in horizontal direction
    D_F_Ver = abs(I[:,0:x-1] - I[:,1:x])
    #variation of the input image (vertical direction)
    D_F_Hor = abs(I[0:y-1,:] - I[1:y,:])
    #variation of the input image (horizontal direction)

    D_B_Ver = abs(B_Ver[:,0:x-1]-B_Ver[:,1:x])
    #variation of the blurred image (vertical direction)
    D_B_Hor = abs(B_Hor[0:y-1,:]-B_Hor[1:y,:])
    #variation of the blurred image (horizontal direction)

    T_Ver = D_F_Ver - D_B_Ver
    #difference between two vertical variations of 2 image (input & blurred)
    T_Hor = D_F_Hor - D_B_Hor
    #difference between two horizontal variations of 2 image (input & blurred)

    V_Ver=np.zeros(T_Ver.shape)
    V_Hor=np.zeros(T_Hor.shape)
    for m in range(T_Ver.shape[0]):
        for n in range(T_Ver.shape[1]):
            V_Ver[m,n] = max(0,T_Ver[m,n])
            V_Hor[n,m] = max(0,T_Hor[n,m])

    S_D_Ver = sum(sum(D_F_Ver[2:y-1,2:x-1]))
    S_D_Hor = sum(sum(D_F_Hor[2:y-1,2:x-1]))

    S_V_Ver = sum(sum(V_Ver[2:y-1,2:x-1]))
    S_V_Hor = sum(sum(V_Hor[2:y-1,2:x-1]))

    blur_F_Ver = (S_D_Ver-S_V_Ver)/S_D_Ver
    blur_F_Hor = (S_D_Hor-S_V_Hor)/S_D_Hor

    blur = max(blur_F_Ver,blur_F_Hor)
    return blur

def exposure(gray_array):
    image = list(im2double(gray_array))

    meanImage = np.mean(image)

    ACMMatrix = [0]*len(image)

    for i in range(len(image)):
        ACMMatrix[i] = abs(image[i]-meanImage)  
    exposureScore = np.mean(ACMMatrix)
    return exposureScore


def GetPS(gray_array,symmetryLineX=60):
    poseSymScore = 0

    image = im2double(gray_array)
    sizePicture = image.shape
    symmetryLineX=int(symmetryLineX)
    if (symmetryLineX-1)>=(sizePicture[0] - symmetryLineX):
        left_half = [(symmetryLineX-(sizePicture[1] - symmetryLineX)),0,sizePicture[1]-symmetryLineX+1,sizePicture[0]]
        right_half = [symmetryLineX-1,0,sizePicture[1],sizePicture[0]]
    else:
        left_half = [0,0,symmetryLineX+1,sizePicture[0]]
        right_half = [symmetryLineX-1,0,sizePicture[1],sizePicture[0]]


    rimage = PIL.Image.fromarray(image)
    image_left = np.array(rimage.crop(left_half))
    image_right =  np.flip(np.array(rimage.crop(right_half)),1)

    s1 = image_left.shape[0]
    s2 = image_right.shape[0]
    padding = abs(s1 - s2)

    if(s1 > s2):
        image_right = np.lib.pad(image_right, (0,padding), 'constant')
    else:
        image_left = np.lib.pad(image_left, (0,padding), 'constant')


    lbp_left   = LBP_3x3(image_left)
    lbp_right  = LBP_3x3(image_right)


    lbp_diff = np.abs(lbp_left-lbp_right)
    poseSymScore = np.sum(lbp_diff)
    return poseSymScore



def GetLS(gray_array,symmetryLineX=60):
    lightSymScore = 0
    image = im2double(gray_array)
    sizePicture = image.shape
    symmetryLineX=int(symmetryLineX)
    if (symmetryLineX-1)>=(sizePicture[0] - symmetryLineX):
        left_half = [(symmetryLineX-(sizePicture[1] - symmetryLineX)),0,sizePicture[1]-symmetryLineX+1,sizePicture[0]]
        right_half = [symmetryLineX-1,0,sizePicture[1],sizePicture[0]]
    else:
        left_half = [0,0,symmetryLineX+1,sizePicture[0]]
        right_half = [symmetryLineX-1,0,sizePicture[1],sizePicture[0]]


    bw = 2
    N = 7
    img_in=image
    img_out_phase = np.zeros((img_in.shape[0],img_in.shape[1],N))
    img_out_mag = np.zeros((img_in.shape[0],img_in.shape[1],N))
    ort = [-45, -30, -15, 0, 15, 30, 45]
    for n in range(0,N):
        img_out_mag[:,:,n],img_out_phase[:,:,n] = imgaborfilt(image=img_in,wavelength=bw, ort=ort[n])
    

    img_out_disp = np.sqrt(np.sum(np.square(img_out_mag), 2))
    img_out_disp = img_out_disp/np.max(img_out_disp)

    rimage = PIL.Image.fromarray(img_out_disp)
    image_left = np.array(rimage.crop(left_half))
    image_right =  np.flip(np.array(rimage.crop(right_half)),1)

    s1 = image_left.shape[0]
    s2 = image_right.shape[0]
    padding = abs(s1 - s2)

    if(s1 > s2):
        image_right = np.lib.pad(image_right, (0,padding), 'constant')
    else:
        image_left = np.lib.pad(image_left, (0,padding), 'constant')


    image_diff = np.abs(image_left-image_right)
    lightSymScore = np.sum(image_diff)
    return lightSymScore

def illuminance_uniformity(gray_array,symmetryLineX=60):
    image = im2double(gray_array)
    sizePicture = image.shape
    symmetryLineX=int(symmetryLineX)
    if (symmetryLineX-1)>=(sizePicture[0] - symmetryLineX):
        left_half = [(symmetryLineX-(sizePicture[1] - symmetryLineX)),0,sizePicture[1]-symmetryLineX+1,sizePicture[0]]
        right_half = [symmetryLineX-1,0,sizePicture[1],sizePicture[0]]
    else:
        left_half = [0,0,symmetryLineX+1,sizePicture[0]]
        right_half = [symmetryLineX-1,0,sizePicture[1],sizePicture[0]]


    rimage = PIL.Image.fromarray(image)
    image_left = np.array(rimage.crop(left_half))
    image_right =  np.flip(np.array(rimage.crop(right_half)),1)

    s1 = image_left.shape[0]
    s2 = image_right.shape[0]
    padding = abs(s1 - s2)

    if(s1 > s2):
        image_right = np.lib.pad(image_right, (0,padding), 'constant')
    else:
        image_left = np.lib.pad(image_left, (0,padding), 'constant')

    S_L = np.std(image_left)
    S_R = np.std(image_right)

    S = (S_L**2)/image_left.size + (S_R**2)/image_right.size

    B_L = np.mean(image_left)
    B_R = np.mean(image_right)

    D = (B_L - B_R)/S

    return D

def GetLS_histogram(gray_array,symmetryLineX=60):
    image = im2double(gray_array)
    sizePicture = image.shape
    symmetryLineX=int(symmetryLineX)
    if (symmetryLineX-1)>=(sizePicture[0] - symmetryLineX):
        left_half = [(symmetryLineX-(sizePicture[1] - symmetryLineX)),0,sizePicture[1]-symmetryLineX+1,sizePicture[0]]
        right_half = [symmetryLineX-1,0,sizePicture[1],sizePicture[0]]
    else:
        left_half = [0,0,symmetryLineX+1,sizePicture[0]]
        right_half = [symmetryLineX-1,0,sizePicture[1],sizePicture[0]]


    rimage = PIL.Image.fromarray(image)
    image_left = np.array(rimage.crop(left_half))
    image_right =  np.flip(np.array(rimage.crop(right_half)),1)

    s1 = image_left.shape[0]
    s2 = image_right.shape[0]
    padding = abs(s1 - s2)

    if(s1 > s2):
        image_right = np.lib.pad(image_right, (0,padding), 'constant')
    else:
        image_left = np.lib.pad(image_left, (0,padding), 'constant')

    hist_left = cv2.calcHist([image_left],[0],None,[256],[0,1])
    hist_right = cv2.calcHist([image_right],[0],None,[256],[0,1])
    
    # E = entropy(pk=hist_left,qk=hist_right)
    E = cv2.compareHist(hist_left, hist_right, cv2.HISTCMP_CHISQR)
    result = None
    # if np.isinf(E[0]) == True:
    #     result=20
    # elif np.isnan(E[0]) == True:
    #     result=20
    # else:
    #     result=E[0]
    result = E
    return result