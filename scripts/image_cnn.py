from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import pandas as pd
import os
import numpy as np
import cv2


def load_data():
    df = pd.read_csv(os.path.dirname(os.getcwd()) + "\data\image_ratings.csv")
    file_nums = set(df['ASIN'])
    base_dir = os.path.dirname(os.getcwd()) + r"\covers\224x224"
    files = os.listdir(base_dir)
    files = [f[:-4] for f in files if f[:-4] in file_nums]
    files.sort()
    df = df[df['ASIN'].isin(files)]
    df.sort_values(by=['ASIN'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    outputImage = np.zeros((64, 64, 3), dtype="uint8")
    images = []
    for f in files:
        outputImage = np.zeros((64, 64, 3), dtype="uint8")
        image = cv2.imread(base_dir+r"\\"+f+".jpg")
        image = cv2.resize(image, (64, 64))
        outputImage[0:64, 0:64] = image
        images.append(outputImage)

    # return our set of images
    return np.array(images), df


def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1

    # define the model input
    inputs = Input(shape=inputShape)

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)

    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model

if __name__=="__main__":
    # load the house images and then scale the pixel intensities to the
    # range [0, 1]
    images, df = load_data()
    images = images / 255.0

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    split = train_test_split(df, images, test_size=0.25, random_state=42)
    (trainAttrX, testAttrX, trainImagesX, testImagesX) = split

    # scale our ratings to the range [0, 1] (will lead to better
    # training and convergence)
    trainY = trainAttrX["rating"] / 5
    testY = testAttrX["rating"] / 5

    # create our Convolutional Neural Network and then compile the model
    # using mean absolute percentage error as our loss, implying that we
    # seek to minimize the absolute percentage difference between our
    # rating *predictions* and the *actual rating*
    model = create_cnn(64, 64, 3, regress=True)
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    # train the model
    print("[INFO] training model...")
    model.fit(trainImagesX, trainY, validation_data=(testImagesX, testY),
              epochs=5, batch_size=8)

    model.save(os.path.dirname(os.getcwd())+r"\models\cover_cnn_model.h5")
    # make predictions on the testing data
    print("[INFO] predicting house prices...")
    preds = model.predict(testImagesX)

    # compute the difference between the *predicted* house prices and the
    # *actual* house prices, then compute the percentage difference and
    # the absolute percentage difference
    diff = preds.flatten() - testY
    percentDiff = (diff / testY) * 100
    absPercentDiff = np.abs(percentDiff)

    # compute the mean and standard deviation of the absolute percentage
    # difference
    mean = np.mean(absPercentDiff)
    std = np.std(absPercentDiff)


    print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))



