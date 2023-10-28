import cv2
from mp_detection import mp_detection
import keras
from preprocessing import tokens
import numpy as np

sequence_length = 1
sequence = []
sentence = []
threshold = 0.8

model = keras.models.load_model('lstm_model.h5')

# Inicializar o vídeo
video_capture = cv2.VideoCapture(0)

colors = [(255,0,0), (0,255,0), (0,0,255)]

def prob_viz(res, tokens, input_frame, colors):
    output_frame = input_frame.copy()
    
    for num, prob in enumerate(res):
        if num < len(colors) and num < len(tokens):
            # Calculate the y-coordinate for the current token
            y = 60 + num * 40
            
            # Draw a filled rectangle to visualize the probability
            cv2.rectangle(output_frame, (0, y), (int(prob * 100), y + 30), colors[num], -1)
            
            # Display the token label
            cv2.putText(output_frame, tokens[num], (0, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return output_frame


while True:
    success, frame = video_capture.read()
    
    if not success:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    landmarks_points, frame = mp_detection(frame_rgb)
    sequence.append(landmarks_points)
    sequence = sequence[-sequence_length:]
    
    if(sequence_length == len(sequence)):
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        
        if res[np.argmax(res)] > threshold: 
            if len(sentence) > 0: 
                if tokens[np.argmax(res)] != sentence[-1]:
                        sentence.append(tokens[np.argmax(res)])
            else:
                sentence.append(tokens[np.argmax(res)])

        if len(sentence) > 5: 
            sentence = sentence[-5:]
    
        frame = prob_viz(res, tokens, frame, colors)

    
    cv2.imshow("Hand Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()