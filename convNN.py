import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.datasets import mnist
import keras
import time
import json
import cv2
from keras.models import model_from_json


def crop(number):
    ret, thresh = cv2.threshold(number, 127, 255, cv2.THRESH_BINARY)
    img2, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contourArr = [cv2.contourArea(c) for c in contours]  # Trazimo najvecu konturu

    if len(contourArr) == 0:
        return number

    contourIndex = np.argmax(contourArr)

    [x, y, w, h] = cv2.boundingRect(contours[contourIndex])

    cropped = number[y:y + h + 1, x:x + w + 1]
    cropped = cv2.resize(cropped, number.shape, interpolation=cv2.INTER_AREA)
    return cropped


def createModel(input_shape, nClasses):
    model = Sequential()
    # The first two layers with 32 filters of window size 3x3
    model.add(Conv2D(28, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(28, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(56, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(56, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))

    return model


if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    #
    # Find the unique numbers from the train labels
    classes = np.unique(train_labels)
    nClasses = len(classes)

    for i in range(len(train_images)):
        cropped = crop(train_images[i])
        train_images[i] = cropped
        cv2.imshow("number" + str(i), train_images[i])
        if cv2.waitKey(1) == 13:
            break

    for i in range(len(test_images)):
        cropped = crop(test_images[i])
        test_images[i] = cropped

    if keras.backend.image_data_format() == 'channels_first':
        nRows, nCols = train_images.shape[1:]
        train_data = train_images.reshape(train_images.shape[0], 1, nRows, nCols)
        test_data = test_images.reshape(test_images.shape[0], 1, nRows, nCols)
        input_shape = (1, nRows, nCols)
    else:
        nRows, nCols = train_images.shape[1:]
        train_data = train_images.reshape(train_images.shape[0], nRows, nCols, 1)
        test_data = test_images.reshape(test_images.shape[0], nRows, nCols, 1)
        input_shape = (nRows, nCols, 1)

    # Change to float datatype
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')

    # Scale the data to lie between 0 to 1
    train_data /= 255

    test_data /= 255

    # Change the labels from integer to categorical data
    train_labels_one_hot = to_categorical(train_labels)
    test_labels_one_hot = to_categorical(test_labels)

    model1 = createModel(input_shape, nClasses)
    # batch_size = 256
    # epochs = 30
    # model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # model1.summary()
    #
    # history = model1.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1,
    #                      validation_data=(test_data, test_labels_one_hot))
    # loss, acc = model1.evaluate(test_data, test_labels_one_hot, verbose=0)
    # model1.save_weights('cnnKerasWeights.h5')
    # # model1.save('kerasCNN')
    # print(acc)

    cropped = crop(test_images[1000]).astype('float32')
    cropped /= 255
    # cv2.imshow("cropped", cropped)
    # if cv2.waitKey(1) == 13:
    #    exit(0)
    model1.load_weights('cnnKerasWeights.h5')
    cropped = np.expand_dims(cropped, axis=0)
    cropped = np.expand_dims(cropped, axis=3)
    number = model1.predict_classes(cropped)

    print(number)