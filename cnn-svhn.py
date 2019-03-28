
from __future__ import absolute_import
from __future__ import print_function
import h5py
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def fetch_data(file_path):
    data = loadmat(file_path)
    return data['X'], data['y']


def convert_to_greyscale(img):
    #Y = 0.2990R + 0.5870G + 0.1140B
    #turn images to greyscale
    return np.expand_dims(np.dot(img, [0.2990, 0.5870, 0.1140]), axis=3)

def preprocessing():
    print("Preprocessing data....")
    #preprocesssing
    train_file_path= 'train_32x32.mat'
    test_file_path= 'test_32x32.mat'

    train_X, train_y = fetch_data(train_file_path)
    test_X, test_y = fetch_data(test_file_path)

    #transposing
    train_X, train_y = train_X.transpose((3,0,1,2)), train_y[:,0]
    test_X, test_y = test_X.transpose((3,0,1,2)), test_y[:,0]

    #change class labels from '10' - '0'
    train_y[train_y == 10] = 0
    test_y[test_y == 10] = 0

    #train test split
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.13, random_state=7)


    #convert to greyscale
    train_gs = convert_to_greyscale(train_X).astype(np.float32)
    test_gs = convert_to_greyscale(test_X).astype(np.float32)
    val_gs = convert_to_greyscale(val_X).astype(np.float32)

    #No longer needed since values are converted to greyscale
    del train_X, test_X, val_X

    ###Normalization
    ## Calculate tmean and std on training data
    trainData_mean = np.mean(train_gs, axis=0)
    trainData_std = np.std(train_gs, axis=0)

    # Calculate difference equally across all the splits
    norm_train_gs = (train_gs - trainData_mean) / trainData_std
    norm_test_gs = (test_gs - trainData_mean)  / trainData_std
    norm_val_gs = (val_gs - trainData_mean) / trainData_std



    e = OneHotEncoder().fit(train_y.reshape(-1, 1))
    train_y = e.transform(train_y.reshape(-1, 1)).toarray()
    test_y = e.transform(test_y.reshape(-1, 1)).toarray()
    val_y = e.transform(val_y.reshape(-1, 1)).toarray()


    #file processing - save grayscale images to file
    outfile = h5py.File('StreetViewHouseNumbersGS.h5', 'w')
    #datasets
    outfile.create_dataset('train_X', data=norm_train_gs)
    outfile.create_dataset('train_y', data=train_y)
    outfile.create_dataset('test_X', data=norm_test_gs)
    outfile.create_dataset('test_y', data=test_y)
    outfile.create_dataset('val_X', data=norm_val_gs)
    outfile.create_dataset('val_y', data=val_y)
    outfile.close()

def model():
    #ConvolutionalLayer #1
    print("Building model....")
    model  = Sequential()
    model.add(Conv2D(32,kernel_size=(5,5), activation='relu', input_shape=(32,32,1)))

    #Pooling Layer #1
    model.add(MaxPooling2D(pool_size=[2, 2], strides=2))

    #Convolutional Layer #2 and Pooling Layer #2
    model.add(Conv2D(64,kernel_size=(5,5),padding="same",activation='relu'))

    model.add(MaxPooling2D(pool_size=[2, 2], strides=2))

    #Dense Layer
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))


    ##dropout
    model.add(Dropout(rate=0.5))

    model.add(Dense(units=10,activation='softmax'))

    ##compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    return model


def main():

    preprocessing()

    in_file = h5py.File('StreetViewHouseNumbersGS.h5', 'r')

    # fetch data
    train_X = in_file['train_X'][:]
    train_y = in_file['train_y'][:]
    test_X = in_file['test_X'][:]
    test_y = in_file['test_y'][:]
    val_X = in_file['val_X'][:]
    val_y = in_file['val_y'][:]
    in_file.close()

    #check to ensure files came in correctly
    print('Training data', train_X.shape, train_y.shape)
    print('Validation data', val_X.shape, val_y.shape)
    print('Test data', test_X.shape, test_y.shape)

    cnn_model = model()

    epochs = 4
    batch = 256

    ##fit model on data
    print("Training....")
    history = cnn_model.fit(train_X, train_y,batch_size=128,epochs=epochs,verbose=1,validation_data=(val_X, val_y))
    print("Saving Model...")
    cnn_model.save('model.h5')

    print("Evaluating Model...")
    metrics = cnn_model.evaluate(test_X, test_y, verbose=0)

    print('Test loss:', metrics[0])
    print('Test accuracy:', metrics[1])

    print(history.history)


main()
