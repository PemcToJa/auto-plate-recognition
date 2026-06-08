import pathlib
from pathlib import Path
import pandas as pd

root_directory = pathlib.Path(__file__).parent.parent.parent

train_images_directory = root_directory / 'data' / 'archive' / 'images' / 'train'
val_images_directory = root_directory / 'data' / 'archive' / 'images' / 'val'
test_images_directory = root_directory / 'data' / 'archive' / 'images' / 'test'

train_labels_directory = root_directory / 'data' / 'archive' / 'labels' / 'train'
val_labels_directory = root_directory / 'data' / 'archive' / 'labels' / 'val'
test_labels_directory = root_directory / 'data' / 'archive' / 'labels' / 'test'

def get_df():
    train_images, val_images, test_images = list(sorted(train_images_directory.glob('*.jpg'))), list(sorted(val_images_directory.glob('*.jpg'))), list(test_images_directory.glob('*.jpg'))
    """
    train_data_pairs loop
    """
    train_data_pairs = []
    for image in train_images:
        with open(train_labels_directory / f"{Path(image).stem}.txt", "r") as file:
            cords = []
            for row in file:
                _, x_center, y_center, width, height = row.split()
                cords.append([x_center, y_center, width, height])
        train_data_pairs.append((train_images_directory / f"{Path(image).stem}.jpg", cords))
    train_df = pd.DataFrame(train_data_pairs, columns=['image_path', 'cords'])
    """
    val_data_pairs loop
    """
    val_data_pairs = []
    for image in val_images:
        with open(val_labels_directory / f"{Path(image).stem}.txt", "r") as file:
            cords = []
            for row in file:
                _, x_center, y_center, width, height = row.split()
                cords.append([x_center, y_center, width, height])
        val_data_pairs.append((val_images_directory / f"{Path(image).stem}.jpg", cords))
    val_df = pd.DataFrame(val_data_pairs, columns=['image_path', 'cords'])
    """
    test_data_pairs loop
    """
    test_data_pairs = []
    for image in test_images:
        with open(test_labels_directory / f"{Path(image).stem}.txt", "r") as file:
            cords = []
            for row in file:
                _, x_center, y_center, width, height = row.split()
                cords.append([x_center, y_center, width, height])
        test_data_pairs.append((test_images_directory / f"{Path(image).stem}.jpg", cords))
    test_df = pd.DataFrame(test_data_pairs, columns=['image_path', 'cords'])

    return train_df, val_df, test_df