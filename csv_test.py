import pandas as pd
import os
import shutil

def organize_images(csv_file, base_image_dir, output_dir):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        print("CSV file read successfully.")
        
        # Print column names to verify
        print("Columns in CSV:", df.columns)
        
        # Delete existing subdirectories in the output_dir
        for sub_dir in os.listdir(output_dir):
            sub_dir_path = os.path.join(output_dir, sub_dir)
            if os.path.isdir(sub_dir_path):
                print(f"Deleting directory: {sub_dir_path}")
                shutil.rmtree(sub_dir_path)
        
        # Create new subdirectories based on labels
        labels = df['ClassId'].unique()
        for label in labels:
            sub_dir_path = os.path.join(output_dir, str(label))
            if not os.path.exists(sub_dir_path):
                print(f"Creating directory: {sub_dir_path}")
                os.makedirs(sub_dir_path)
        
        # Move images to the appropriate subdirectories
        for index, row in df.iterrows():
            # Convert path to string and strip
            image_path = str(row['Path']).strip()  # Path of the image relative to base_image_dir
            class_label = str(row['ClassId'])  # Label used to determine the target directory
            
            # Remove 'Test/' from the beginning of the path if it exists
            if image_path.startswith('Test/'):
                image_path = image_path[len('Test/'):]
            
            # Construct the full image path by combining the base directory and the relative path
            full_image_path = os.path.join(base_image_dir, 'Test', image_path)
            print(f"Constructed full path: {full_image_path}")
            
            # Check if the image file exists
            if not os.path.isfile(full_image_path):
                print(f"Image file not found: {full_image_path}")
                continue
            
            # Construct the destination directory based on the class label
            dest_dir = os.path.join(output_dir, class_label)
            
            # Construct the destination path for the image
            dest_path = os.path.join(dest_dir, os.path.basename(image_path))
            
            # Move the image to the destination directory
            print(f"Moving {full_image_path} to {dest_path}")
            shutil.move(full_image_path, dest_path)
        
        print("Image organization completed.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
csv_file = "/Users/jakehopkins/Downloads/German Traffic Signs Dataset/Test.csv"
base_image_dir = "/Users/jakehopkins/Downloads/German Traffic Signs Dataset"
output_dir = "/Users/jakehopkins/Downloads/German Traffic Signs Dataset/Test"
organize_images(csv_file, base_image_dir, output_dir)
