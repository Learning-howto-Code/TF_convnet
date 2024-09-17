import os
import glob
from sklearn.model_selection import train_test_split
import shutil
import csv
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from my_utils import split_data

from deeplearning_models import streetsigns_model


def order_test_set(path_to_images, path_to_csv):
    
    # Check if the CSV file exists
    if os.path.exists(path_to_csv):
        print("File exists")
    else:
        print("File does not exist")
        return  # Exit the function if the file does not exist

    testset = {}

    try:
        with open(path_to_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            for i, row in enumerate(reader):
                if i == 0:
                    continue  # Skip the header row

                img_name = row[-1].replace('Test/', '')
                label = row[-2]     

                path_to_folder = os.path.join(path_to_images, label)

                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)

                img_full_path = os.path.join(path_to_images, img_name)
                if os.path.exists(img_full_path):
                    shutil.move(img_full_path, path_to_folder)
                else:
                    print(f"[Warning]: Image {img_name} not found in {path_to_images}")

    except Exception as e:
        print(f'[Info]: Error reading CSV file: {e}')

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
epochs = 15

train_generator, val_generator, train_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
nbr_classes = train_generator.num_classses

path_to_save_model = './Models'
ckpt_saver = ModelCheckpoint(
    path_to_save_model,
    monitor="val_accuracy"
    mode='max',
    save_best_only=True,
    save_freq='epoch',
    verbose=1
)

early_stop = EarlyStopping(monitor="val_accuracy", patience=10)

model = streetsigns_model(nbr_classes)

model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics= ['accuracy'])

model.fit(train_generator)
    epochs= epochs,
    batch_size=batch_size,
    validation_data=val_generator
    callbacks=[ckpt_saver,early_stop]
)
