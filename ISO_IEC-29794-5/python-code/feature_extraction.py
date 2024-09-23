import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import PIL.Image
from GetQualityMetric import GetQualityMetric
import os
from matlab_functions import rgb2gray
import cv2, glob
import argparse
import pandas as pd
from pathlib import Path


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Assess the face image quality')
    parser.add_argument('-i','--img_path', default='/Users/soler/DATABASES/HAND/HaGRID/Quality/Images/', help='path to the training folder subsamples', type=str)
    parser.add_argument('-m','--mask_path', default='/Users/soler/DATABASES/HAND/HaGRID/Quality/Masks/', help='path to the segmentation maps of the training folder subsamples', type=str)
    parser.add_argument('-s','--saving_path', default='results/', help='path for storing the features',type=str)

    
    # Get config from arguments
    args = parser.parse_args()

    #path (only) of images to be assessed
    path = args.img_path
    mask_path = args.mask_path

    #path for storing results
    saving_path=args.saving_path
    os.makedirs(saving_path,exist_ok=True)


    file_list = list(glob.glob(os.path.join(path, "*/**/*.jpg")))


    categories = {'0-24': 0,
                  '25-49': 1,
                  '50-74': 2,
                  '75-100': 3}

    cnt = 0
    x_train = []
    y_train = []
    for image_path in file_list:
        cropped_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask_name = os.path.join(mask_path, Path(image_path).parent.parent.name, Path(image_path).parent.name, "{}.png".format(Path(image_path).stem))
        mask_img = cv2.imread(mask_name, cv2.IMREAD_COLOR)
        rgb = PIL.Image.fromarray(cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB)).resize((324, 324),PIL.Image.ANTIALIAS)
        mask_img = PIL.Image.fromarray(cv2.cvtColor(mask_img,cv2.COLOR_BGR2RGB)).resize((324, 324),PIL.Image.ANTIALIAS) 
        rgb_array=np.array(rgb)
        mask_img = rgb2gray(np.array(mask_img))
        mask_img= cv2.dilate(mask_img, np.ones((5, 5), np.uint8))  
        Qmetrics = [GetQualityMetric(rgb_array, mask_img)]
        x_train.append(Qmetrics)
        y_train.append(categories[Path(image_path).parent.parent.name])

        cnt = cnt+1
        print('%d/%d' % (cnt, len(file_list)), end='\r')

    x_train = np.array(x_train)
    x_train = np.squeeze(x_train, axis=1)

    np.save("xtrain.npy", x_train)
    np.save("ytrain.npy", y_train)

    print('Feature extraction done...')

if __name__ == "__main__":
    main()