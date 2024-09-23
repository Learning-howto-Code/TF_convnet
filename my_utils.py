import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import shutil
import csv, pdb
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#comment for git
def display_some_examples(examples, labels):

    
    plt.figure(figsize=(10, 10))
    
    for i in range(25):
        idx = np.random.randint(0, examples.shape[0] - 1)
        img = examples[idx]
        label = labels[idx]
        plt.subplot(5, 5, i + 1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')
    plt.show()


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

                img_name = row[-1].replace('Test/', '').replace('Train/', '')  # Remove any 'Test/' or 'Train/' prefix
                label = row[-2]

                path_to_folder = os.path.join(path_to_images, label)
                pdb.set_trace()
                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)

                img_full_path = os.path.join(path_to_images, img_name)

                # Debugging: print the paths being checked
                print(f"Looking for image: {img_full_path}")

                if os.path.exists(img_full_path):
                    shutil.move(img_full_path, path_to_folder)
                else:
                    print(f"[Warning]: Image {img_name} not found in {path_to_images}")

    except Exception as e:
        print(f'[Info]: Error reading CSV file: {e}')





def split_data(path_to_data, path_to_save_train, path_to_save_val, split_size=0.1):
    folders = os.listdir(path_to_data)
    print(f"Folders found: {folders}")

    for folder in folders:

        full_path = os.path.join(path_to_data, folder)
        images_paths = glob.glob(os.path.join(full_path, '*.png'))
        print(f"Processing folder: {folder}, found {len(images_paths)} images")

        x_train, x_val = train_test_split(images_paths, test_size=split_size)

        for x in x_train:

            basename = os.path.basename(x)
            path_to_folder = os.path.join(path_to_save_train, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)

        for x in x_val:

            path_to_folder = os.path.join(path_to_save_val, folder)
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)

def create_generators(batch_size, train_data_path, val_data_path, test_data_path):

    train_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.,
        rotation_range=10,
        width_shift_range=0.1
    )

    test_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.,
    )

    train_generator = train_preprocessor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = test_preprocessor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size,
    )
# test gnerator not finding any images
    test_generator = test_preprocessor.flow_from_directory(
        test_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size,
    )

    return train_generator, val_generator, test_generator


# if __name__ == '__main__':

#     if False:
#         path_to_data = "/Users/jakehopkins/Downloads/German Traffic Signs Dataset/Train"
#         path_to_save_train = "/Users/jakehopkins/Downloads/German Traffic Signs Dataset/training_data/train"
#         path_to_save_val = "/Users/jakehopkins/Downloads/German Traffic Signs Dataset/training_data/val"
#         split_data(path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)

#     # Define the paths to the images and the CSV file
#     path_to_images = "/Users/jakehopkins/Downloads/German Traffic Signs Dataset/Train"
#     path_to_csv = '/Users/jakehopkins/Downloads/German Traffic Signs Dataset/Train.csv'
    
#     # Call the function to organize the test set
#     order_test_set(path_to_images, path_to_csv)
            