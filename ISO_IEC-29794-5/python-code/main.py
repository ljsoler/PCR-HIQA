import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import PIL.Image
from GetQualityMetric import GetQualityMetric
import os
from matlab_functions import rgb2gray
import cv2
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Compute hand image quality scores based on an adaptation of ISO/IEC 29794-5')
    parser.add_argument('-i','--img_path', required=True, default="test_images",
                        help='path (only) of images to be assessed', type=str)
    parser.add_argument('-m','--mask_path', required=True, default="test_masks",
                        help='path to the respective masks o the images to be assessed', type=str)
    parser.add_argument('-f','--saving_file', default='results/test_examples.csv', 
                        help='path to save the prediction scores',type=str)

    
    # Get config from arguments
    args = parser.parse_args()

    #path (only) of images to be assessed
    path = args.img_path
    mask_path = args.mask_path

    #path for storing results
    saving_path=args.saving_file

    x_train = np.load('xtrain.npy')
    y_train = np.load('ytrain.npy')

    indices = np.arange(len(y_train))
    random.shuffle(indices)

    x_train = x_train[indices]
    y_train = y_train[indices]

    regr = RandomForestRegressor(max_depth=16, n_estimators=109, random_state=456)
    regr.fit(x_train, y_train)

    file_list = list(Path(path).glob("*.[jp][pn][g]")) 

    data = {}
    failure = []
    print('Prediction starts...')
    for image_path in tqdm(file_list):
        try:
            cropped_img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask_name = os.path.join(mask_path, "{}.png".format(image_path.stem))
            mask_img = cv2.imread(mask_name, cv2.IMREAD_COLOR)
            rgb = PIL.Image.fromarray(cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB)).resize((324, 324),PIL.Image.LANCZOS)
            mask_img = PIL.Image.fromarray(cv2.cvtColor(mask_img,cv2.COLOR_BGR2RGB)).resize((324, 324),PIL.Image.LANCZOS) 
            rgb_array=np.array(rgb)
            mask_img = rgb2gray(np.array(mask_img))
            mask_img= cv2.dilate(mask_img, np.ones((5, 5), np.uint8))  
            Qmetrics = GetQualityMetric(rgb_array, mask_img)
            score = regr.predict([Qmetrics])[0]
            key = '{}'.format(image_path.name)
            data[key] = score
        except:
            failure.append(image_path)
        
    print('Prediction done...')

    print("Printing failure cases...")
    for f in failure:
        print(f)

    #store quality scores
    headers = ['filename','score']
    results = pd.DataFrame(columns=headers, data = data.items())
    print(results.head())
    results.to_csv(saving_path, index=None)

    print('Results stored...')

if __name__ == "__main__":
    main()