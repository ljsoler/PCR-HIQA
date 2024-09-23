# hand-image-quality-ISO-IEC-29794-5-python used in PCR-HIQA: Perceptual Classifiability Ratio for Hand Image Quality Assessment

This project contains the implementation of the ISO/IEC 29794-5 metrics adapted for hand image quality in PCR-HIQA. It is based on the [ISO/IEC 29794-5](https://share.nbl.nislab.no/g03-03-sample-quality/face-image-quality) implementation for face image quality.

# How to run the script

## 1. Install requirements 
Python 3.9
pip install -r requirements.txt

## 2. Run the script 
python main.py 

The default setting is:

python main.py --img_path test_images/ --mask_path test_masks/ --saving_file results/

# Arguments

--img_path: path (only) of images to be assessed

--mask_path: path to the respective masks o the images to be assessed

--saving_file: path to save the prediction scores in form of a csv file.

## In the "results" folder, you can find the hand image quality scores for HaGrid training and test sets  
