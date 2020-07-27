# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from lr_utils import load_dataset
print(tf.__version__)
train_images , trainSet_y , test_images , testSet_y , classes = load_dataset()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    # keras.layers.Flatten(input_shape=(64, 64, 3)),
    keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(64, 64, 3)),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(64, 64, 3)),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(64, 64, 3)),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu',),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(1,activation='linear')
])

model.summary()

model.compile(optimizer=RMSprop(lr=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, trainSet_y.T,steps_per_epoch=10, epochs=100)

test_loss, test_acc = model.evaluate(test_images,  testSet_y.T, verbose=2)

print('\nTest accuracy:', test_acc)
