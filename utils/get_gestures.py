import json
import cv2
import sys
import getopt
from pathlib import Path
import os
import numpy as np

def get_hand_area(image, landmarks):
    image_height = image.shape[0]
    image_width = image.shape[1]

    annotated_image = np.zeros_like(image)

    landmarks_transf = np.array(landmarks)*[image_width, image_height]
    bounding_box = [[landmarks_transf[0][0] + 30, landmarks_transf[0][1]], [landmarks_transf[13][0] + 15, landmarks_transf[13][1] - 25], [landmarks_transf[10][0] - 10, landmarks_transf[10][1] - 10], [landmarks_transf[3][0] - 5, landmarks_transf[3][1] - 5], [landmarks_transf[0][0] - 40, landmarks_transf[0][1]]] #for mute
    # bounding_box = [landmarks_transf[0], landmarks_transf[1], landmarks_transf[2], landmarks_transf[5], landmarks_transf[9], landmarks_transf[13], landmarks_transf[17]]
    cv2.fillPoly(annotated_image, np.int32([bounding_box]), (255, 255, 255))

    cropped_image = np.zeros_like(image)
    cropped_image[np.where(annotated_image == (255, 255, 255))] = image[np.where(annotated_image == (255, 255, 255))]

    return cropped_image

def extract_gestures(images_folder, annotations_folder, output_folder, landmark_folder, gesture, verbose = False):
    
    os.makedirs(output_folder, exist_ok=True)
    if(verbose):
        print("Processing gesture: " + gesture)
    index = 0
    with open(annotations_folder + '/' + gesture + '.json') as json_file:
        data = json.load(json_file)
        images_count = len(data.items())
        if(verbose):
            print(images_count)
        for key, value in data.items():
            try:
                if verbose and index % 100 == 0:
                    print("Processing image " + str(index) + " out of " + str(images_count)) 
                # print(key)
                gesture_index = value["labels"].index(gesture)
                bbox = value["bboxes"][gesture_index]
                X = bbox[0]
                Y = bbox[1]
                W = bbox[2]
                H = bbox[3]
                hand = value["leading_hand"] #Only in the first version

                id = value["user_id"]

                if len(value['landmarks']) > 0 and len(value['landmarks'][gesture_index]) > 0: #Only in the first version

                    image_path = os.path.join(images_folder, gesture, key + '.jpg')
                    gesture_image = cv2.imread(image_path)
                    #print(gesture_image) 
                    image_height = gesture_image.shape[0]
                    image_width = gesture_image.shape[1]
                    X = int(round(X * image_width))
                    Y = int(round(Y * image_height))
                    W = int(round(W * image_width))
                    H = int(round(H * image_height))

                    X_c = X + int(W/2)
                    Y_c = Y + int(H/2)

                    square_side = int(max(W, H)/2)
                    bordered_image=cv2.copyMakeBorder(gesture_image,square_side,square_side,square_side,square_side,cv2.BORDER_CONSTANT, value=[0,0,0])

                    cropped_image = bordered_image[Y_c: Y_c + square_side*2, X_c: X_c + square_side*2]

                    output_path = os.path.join(output_folder, id, gesture, hand)
                    os.makedirs(output_path, exist_ok=True)
                    cv2.imwrite(os.path.join(output_path, key + '.jpg'), cropped_image)

                    #compute metrics
                    # hand_area = get_hand_area(gesture_image, value['landmarks'][gesture_index])
                    # bordered_image=cv2.copyMakeBorder(hand_area,square_side,square_side,square_side,square_side,cv2.BORDER_CONSTANT, value=[0,0,0])
                    # cropped_image = bordered_image[Y_c: Y_c + square_side*2, X_c: X_c + square_side*2]
                    # cv2.imwrite(os.path.join(output_path, key + '_area.jpg'), hand_area)

                    landmarks = []
                    #Get landmarks in the spatial coordenates
                    for x, y in value['landmarks'][gesture_index]:
                        landmarks.append((int(x*image_width + square_side - X_c), int(y*image_height + square_side - Y_c)))

                    os.makedirs(landmark_folder, exist_ok=True)

                    data_json = {'id': id,
                                 'hand': hand,
                                 'gesture': gesture,
                                 'landmarks': landmarks}
                    
                    with open(os.path.join(landmark_folder, '{}.json'.format(key)), "w") as write_file:
                        json.dump(data_json, write_file)


            except Exception as e:
                print("The error is: {}".format(e))
                print(value['landmarks'])
                print(value["labels"])

            index += 1

if __name__ == "__main__":
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv,"m:q:i:o:l:a:", ['verbose'])


    images_folder = "/media/datasets/HaGRID/data/train"
    annotations_folder = "/media/datasets/HaGRID/data/train/ann_train_val"
    output_folder = "/home/fbi1532/Databases/HAND/HaGRID/Images/train"
    landmark_folder = "/home/fbi1532/Databases/HAND/HaGRID/Info/train"
    verbose = True

    for opt, arg in opts:
        if opt == '-i':
            images_folder = arg
        elif opt == '-a':
            annotations_folder = arg
        elif opt == '-l':
            landmark_folder = arg
        elif opt == '-o':
            output_folder = arg
        elif opt == '--verbose':
            verbose = True
        else:
            print(f"Incorrect parameter: {opt}")
            sys.exit()
    
    gestures_list = ['call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted']

    for g in gestures_list:
        extract_gestures(images_folder, annotations_folder, output_folder, landmark_folder, g, verbose)

    #image_folder se refiere a la carpeta en la que se encuentran las imagenes,
    #annotations_folder se refiere a las coordenadas en las que se recortan las imagenes
    #output_folder se refiere al fichero en el que se van a guardar las soluciones este fichero es creado por la funcion
