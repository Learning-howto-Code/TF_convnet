import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Flatten
from tensorflow.keras import Model


# Functional approach: function that returns a model
def functional_model():
    my_input = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(my_input)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=my_input, outputs=x)
    return model

# Tensorflow.keras.Model : inherit from this class
class MyCustomModel(tf.keras.Model):
        
    def __init__(self, K1, K2):
        super().__init__()
        self.conv1 = Conv2D(32, K1, activation='relu')# passing in K1 isntead of 3,3
        self.conv2 = Conv2D(64, K2, activation='relu')
        self.maxpool1 = MaxPooling2D()
        self.batchnorm1 = BatchNormalization()

        self.conv3 = Conv2D(128, (3,3), activation='relu')
        self.maxpool2 = MaxPooling2D()
        self.batchnorm2 = BatchNormalization()

        self.GlobalAvgpool1 = GlobalAveragePooling2D()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='relu')


    def call(self, my_input):

        x = self.conv1(my_input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        x = self.GlobalAvgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x
    

def streetsigns_model(nbr_classes):

    my_input = Input(shape=(60,60,3))

    x = Conv2D(32, (3, 3), activation='relu')(my_input)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)

    # x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(nbr_classes, activation='relu')(x)

    return Model(inputs=my_input, outputs=x)

if __name__ =='__main__':

    model = streetsigns_model(10)
    model.summary()


