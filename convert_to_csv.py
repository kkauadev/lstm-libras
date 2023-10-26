import os
import string
import numpy as np

import cv2
from mp_detection import mp_detection

# Constantes com letras do alfabeto
ALPHABET = list(string.ascii_uppercase)

# Diretórios de entrada e saída
DIR_CSV_OUTPUT = './data'
DIR_CSV_INPUT = './dataset'

def read_image(image_path):
    frame = cv2.imread(image_path)

    if frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks_points, frame = mp_detection(frame_rgb)
        return landmarks_points

def read_video(image_path):
    frame_landmarks = []
    video_capture = cv2.VideoCapture(image_path)
    
    while True:
        success, frame = video_capture.read()
        
        if not success:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks_points, frame = mp_detection(frame_rgb)
        frame_landmarks.append(landmarks_points)
        
    video_capture.release()
    return frame_landmarks

def create_data_training(base_directory):
    existing_folders = [f for f in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, f))]

    if not existing_folders:
        next_folder_name = os.path.join(base_directory, "0")
    else:
        last_folder = max(existing_folders, key=int)
        next_folder_number = int(last_folder) + 1
        next_folder_name = os.path.join(base_directory, str(next_folder_number))
    
    os.mkdir(next_folder_name)
    return next_folder_name
# Iterar sobre cada letra do alfabeto
for word_dir_name in os.listdir(DIR_CSV_INPUT):
# for word_dir_name in ['A']:
    input_folder = DIR_CSV_INPUT + '/' + word_dir_name
    output_folder = DIR_CSV_OUTPUT + "/" + word_dir_name
        
    # Verifica se a pasta de saída existe e, se não, criá-la
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        
    # Iterar sobre as imagens na pasta de entrada
    for image_filename in os.listdir(input_folder):
    # for image_filename in ["1.jpg", "2.jpg"]:
        hand_data = []
        
        image_path = os.path.join(input_folder, image_filename)
        print(image_path)
        
        file_type = os.path.splitext(image_filename)[1]
        
        # if file_type == ".jpg" or ".png":
        if file_type == ".png":
            landmarks_points = read_image(image_path)
            hand_data.append(landmarks_points)  
            
        elif file_type == ".mp4":
            hand_data = read_video(image_path)
        
        data_training_path = create_data_training(output_folder)
        
        for idx, hand_landmarks in enumerate(hand_data):
            np.save(f"{data_training_path}/{idx}.npy", hand_landmarks)
