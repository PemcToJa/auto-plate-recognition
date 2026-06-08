import cv2
import torch
from torch.utils.data import Dataset

class LicensePlateObjectDetectionDataset(Dataset):
    def __init__(self, dataframe=None, split_size=None, transform=None):
        self.df = dataframe
        self.split_size = split_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_bgr = cv2.imread(row['image_path'])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        cords_list = row['cords']
        labels = [0] * len(cords_list)

        if self.transform:
            augmented = self.transform(image=image_rgb, bboxes=cords_list, class_labels=labels)
            image = augmented['image']
            cords_list = augmented['bboxes']
        else: return None, None

        target = torch.zeros((self.split_size, self.split_size, 5))

        for box in cords_list:
            x_fraction, y_fraction, w_fraction, h_fraction = map(float, box)

            x_fraction = max(0.0, min(x_fraction, 0.9999))
            y_fraction = max(0.0, min(y_fraction, 0.9999))

            """
            Tutaj sprawdzamy do jakiego wiersza (grid_row) i której kolumny (grid_column) siatki 
            wpada środek tablicy. Robimy to tak że na wybieramy minimalnego inta z 
            'self.split_size * x_fraction' ponieważ 'x_fraction' - to ułamek podzielony przez 
            szerokość obrazu(podobnie z y_fraction) to wtedy pomnożone przez 14 da nam wiedzę 
            gdzie dokładnie teń srodek obiektu wypadnie. Dalej mamy zabezpieczneie w tym 'min()'
            czyli 'self.split_size - 1' to oznacza że nawet jeżeli w 'int(self.split_size * x_fraction)'
            wyjdzie idealnie 14 co wyawliło by błąd typu out of bounds error wybralibyśmy 13
            """
            grid_column, grid_row = min(int(self.split_size * x_fraction), self.split_size - 1), min(int(self.split_size * y_fraction), self.split_size - 1)

            """
            Tutaj natomiast obliczamy pozycję wewnątrz komórki robiąc np.: 
            '(self.split_size * x_fraction) - grid_column' Mnożymy przez siatkę: 14 * 0.53 = 7.42
            a potem odejmujemy 'grid_column' czyli dostajemy '0.42' czyli pozycję wewnątrz komórki. 
            """
            x_position_inside_cell, y_position_inside_cell = (self.split_size * x_fraction) - grid_column, (self.split_size * y_fraction) - grid_row

            if target[grid_row, grid_column, 0] == 0:
                target[grid_row, grid_column, 0] = 1.0
                target[grid_row, grid_column, 1] = x_position_inside_cell
                target[grid_row, grid_column, 2] = y_position_inside_cell
                target[grid_row, grid_column, 3] = w_fraction
                target[grid_row, grid_column, 4] = h_fraction

        return image, target