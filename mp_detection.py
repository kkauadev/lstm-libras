import numpy as np
import mediapipe as mp

mp_draw_landmarks = mp.solutions.drawing_utils.draw_landmarks
hand_connections = mp.solutions.hands.HAND_CONNECTIONS
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands()

# num_coordinates = NUM_COORDINATES
num_coordinates = 3
max_landmarks = 7

def mp_detection(frame):
    landmarks_points_left = np.zeros((max_landmarks * num_coordinates,))
    landmarks_points_right = np.zeros((max_landmarks * num_coordinates,))
        
    # Process hands on frame
    hands_results = hands_detector.process(frame)
    
    # draw hands landmarks
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_draw_landmarks(frame, hand_landmarks, hand_connections)
            
            left_index = 0
            right_index = 0
            
             # Separar dados das mãos esquerda e direita                        
            for landmark in hand_landmarks.landmark:
                x, y, z = landmark.x, landmark.y, landmark.z
                if x < 0.5 and left_index < max_landmarks * num_coordinates:
                    landmarks_points_left[left_index] = x
                    landmarks_points_left[left_index + 1] = y
                    landmarks_points_left[left_index + 2] = z
                    left_index += num_coordinates
                elif right_index < max_landmarks * num_coordinates:
                    landmarks_points_right[right_index] = x
                    landmarks_points_right[right_index + 1] = y
                    landmarks_points_right[right_index + 2] = z
                    right_index += num_coordinates
                    
    # todos os pontos das mãos são conectados em ordem, cada mão com 21 pontos, os 21 primeiros são mãos esquerda e o resto direita
    hand_landmarks = np.concatenate((landmarks_points_left, landmarks_points_right))
    
    return hand_landmarks, frame