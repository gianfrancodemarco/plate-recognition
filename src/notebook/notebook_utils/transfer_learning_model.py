from src.datasets.DatasetLoader import DatasetLoader
from matplotlib import pyplot as plt
import logging
import cv2
from src.datasets.AnnotatedImageVisualizer import AnnotatedImageVisualizer

from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

models_path = '/content/gdrive/MyDrive/ml_models/'
IMAGE_SIZE = 256
current_epochs = 0
epochs = 10000
save_every_n_epochs = 50
base_model_name = 'MSE_EfficientNetV2M'

if base_model_name == 'VGG16':
    from keras.applications.vgg16 import VGG16

    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
elif base_model_name == 'RMSE_EfficientNetV2M' or base_model_name == 'MSE_EfficientNetV2M':
    from keras.applications.efficientnet_v2 import EfficientNetV2M

    base_model = EfficientNetV2M(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

try:
    print("Loading model")

    models = os.listdir(models_path)
    models = [model for model in models if not model.startswith('.')]
    models = [model for model in models if base_model_name in model]
    models.sort(key=lambda el: int(el.split('_')[-1].split('.')[0]))  # sort by number of epochs
    last_model = models[-1]
    current_epochs = int(last_model.split('_')[-1].split('.')[0])
    model = load_model(os.path.join(models_path, last_model))

    print(f"Loaded model {last_model}")

except Exception as e:

    print(e)
    print("Creating model")

    model = Sequential()
    model.add(base_model)

    model.layers[-1].trainable = False

    model.add(Conv2D(10, 2, activation='relu', padding='same'))
    model.add(Conv2D(10, 3, activation='relu', padding='same'))
    model.add(Conv2D(10, 4, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(10, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.summary()
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

while current_epochs < epochs:
    current_epochs += save_every_n_epochs

    model_name = f'plate_detection_hitl_{base_model_name}_{current_epochs}.h5'

    print(f"Training the model for {save_every_n_epochs} epochs")

    train = model.fit(X, y, epochs=save_every_n_epochs, batch_size=16, verbose=1)
    print("Model trained")

    print("Saving the model")
    model.save(os.path.join(models_path, model_name))
    print(f"Saved model {model_name}")

# Test
# scores = model.evaluate(X_test, y_test, verbose=1)
# print("Score : %.2f%%" % (scores[1] * 100))