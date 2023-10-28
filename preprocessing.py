from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import os

tokens = os.listdir("./data")

classes = {label:num for num, label in enumerate(tokens)}

sequences, labels = [], []

for token in tokens:
# for token in ["A"]:
    for tokens_data in os.listdir("./data/" + token):
    # for tokens_data in ["0", "1", "2"]:
        frame_sequence = []
        
        for npy_file in os.listdir("./data/" + token + "/" + tokens_data):
            res = np.load(os.path.join("./data/" + token + "/" + tokens_data, npy_file))
            frame_sequence.append(res)
        
        if len(frame_sequence) > 0:
            sequences.append(frame_sequence)
            labels.append(classes[token])
        
X = np.array(sequences)
y = to_categorical(labels).astype(int)

preprocesseded_data = train_test_split(X, y, test_size=0.2)