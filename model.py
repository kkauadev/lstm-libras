from preprocessing import preprocesseded_data

from keras.models import Sequential
from keras.layers import LSTM, Dense

X_train, X_test, y_train, y_test = preprocesseded_data

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(1, 42)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# esse 22 representa o valor total de todos os tokens que podem ser encontrados, sendo que cada um deles é um número inteiro
# se aumentar o numero de tokens mude esse valor
model.add(Dense(22, activation='softmax'))

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=100)

model.save("lstm_model.h5")