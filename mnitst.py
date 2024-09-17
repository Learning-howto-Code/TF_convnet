import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from deeplearning_models import functional_model, MyCustomModel
from my_utils import display_some_examples

# Tensorflow.keras.Sequential
seq_model = tf.keras.Sequential(
    [
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ]
)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape) 
    print("x_test.shape = ", x_test.shape)
    print("y_test.shape = ", y_test.shape)

    if False:
        display_some_examples(x_train, y_train)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # model = functional_model()
    model = MyCustomModel((4,4),(5,5))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
    # Model Training
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2, verbose = 1)

    # Evaluation on test set
    model.evaluate(x_test, y_test, batch_size=64)


