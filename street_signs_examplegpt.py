import os
import glob
from sklearn.model_selection import train_test_split
import shutil
import csv
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from my_utils import create_generators, split_data
from deeplearning_models import streetsigns_model
import pdb



if __name__ == '__main__':

    if False:
        path_to_data = "/Users/jakehopkins/Downloads/German Traffic Signs Dataset/Train"
        path_to_save_train = "/Users/jakehopkins/Downloads/German Traffic Signs Dataset/training_data/train"
        path_to_save_val = "/Users/jakehopkins/Downloads/German Traffic Signs Dataset/training_data/val"
        split_data(path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)

        # Define the paths to the images and the CSV file
        path_to_images = "/Users/jakehopkins/Downloads/German Traffic Signs Dataset/Train"
        path_to_csv = '/Users/jakehopkins/Downloads/German Traffic Signs Dataset/Train.csv'

        # Call the function to organize the test set
        order_test_set(path_to_images, path_to_csv)

    path_to_train = '/Users/jakehopkins/Downloads/German Traffic Signs Dataset/training_data/train'
    path_to_val = '/Users/jakehopkins/Downloads/German Traffic Signs Dataset/training_data/val'
    path_to_test = '/Users/jakehopkins/Downloads/German Traffic Signs Dataset/Test'
    batch_size = 64
    epochs = 3

    # pdb.set_trace()
    # Create data generators
    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes

    path_to_save_model = './Models'
    if not os.path.exists(path_to_save_model):
        os.makedirs(path_to_save_model)

    ckpt_saver = ModelCheckpoint(
        os.path.join(path_to_save_model, 'best_model.keras'),
        monitor="val_accuracy",
        mode='max',
        save_best_only=True,
        save_freq='epoch',
        verbose=1
    )

    early_stop = EarlyStopping(monitor="val_accuracy", patience=10)

    model = streetsigns_model(nbr_classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[ckpt_saver, early_stop]
    )