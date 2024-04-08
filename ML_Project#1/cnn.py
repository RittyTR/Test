# Suhaila Kondappilly Aliyar,fdai7995,1492822
# Ritty Tharakkal Raphel,fdai7690,1459915 

# Command to execute script
# python3 cnn.py <npz> imgW imgH imgC train|test

import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers.legacy import Adam as Adam_legacy
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical

def load_data(npz_file):
    data = np.load(npz_file)
    images = data['images']
    labels = data['labels']
    return images, labels

def preprocess_data(images, labels):
    images = images / 255.0
    images = images.reshape(-1, imgW, imgH, imgC)
    labels = to_categorical(labels, num_classes=10)
    return images, labels

def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(imgW, imgH, imgC)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=Adam_legacy(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32, validation_data=None):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
    return model

def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

def save_weights(model, weights_file):
    model.save_weights(weights_file)

def load_weights(model, weights_file):
    model.load_weights(weights_file)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 cnn.py <npz> imgW imgH imgC train|test")
        sys.exit(1)

    npz_file = sys.argv[1]
    imgW = int(sys.argv[2])
    imgH = int(sys.argv[3])
    imgC = int(sys.argv[4])
    train_test = sys.argv[5]

    images, labels = load_data(npz_file)
    images, labels = preprocess_data(images, labels)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train, y_train = shuffle(X_train, y_train)

    model = create_model()

    if train_test == "train":
        model = train_model(model, X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        save_weights(model, 'model_weights.h5')
    elif train_test == "test":
        load_weights(model, 'model_weights.h5')
        evaluate_model(model, X_test, y_test)
    else:
        print("Invalid command. Use 'train' or 'test'.")
