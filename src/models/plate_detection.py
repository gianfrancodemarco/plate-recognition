import os

from src.datasets.DatasetLoader import DatasetLoader
import numpy as np
from matplotlib import pyplot as plt
import logging
import cv2

logger = logging.getLogger(__name__)

IMAGE_SIZE = 256

dl = DatasetLoader()
dl.load_kaggle_dataset()

X = dl.kaggle_X
y = dl.kaggle_y

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.applications.vgg16 import VGG16

try:
    logger.info("Loading model")
    #model = load_model(os.path.join("models", "my_model.h5"))
    model = load_model("my_model.h5")
    logger.info("Loaded model")
except Exception as e:
    logger.warning(e)

    logger.info("Creating model")
    # Create the model
    model = Sequential()
    model.add(VGG16(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.layers[-6].trainable = False
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    logger.info("Training the model")
    train = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=32, verbose=1)
    logger.info("Model trained")

    logger.info("Saving the model")
    model.save('my_model.h5', overwrite=True)
    logger.info("Model saved")

# Test
scores = model.evaluate(X_test, y_test, verbose=1)
print("Score : %.2f%%" % (scores[1] * 100))

y_cnn = model.predict(X_test)

plt.figure()
for i in range(0, 43):
    plt.axis('off')

    ny = y_cnn[i]
    X_test[i] = X_test[i]

    image = cv2.rectangle(X_test[i],(int(ny[0]),int(ny[1])),(int(ny[2]),int(ny[3])),(0, 255, 0))
    plt.imshow(image)
    plt.show()