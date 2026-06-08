import pathlib
import matplotlib.pyplot as plt
from torch import optim
import torch
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.data_curation.dataframe_assembler import get_df
from src.dataset_construction.dataset import LicensePlateObjectDetectionDataset
from src.model_construction.PlateLocNet_Classifier import PlateLocNet
from src.model_training.loss_calculation.loss import get_loss

root_directory = pathlib.Path(__file__).parent.parent.parent
charts_save_path = root_directory / 'src' / 'models' / 'charts' / 'PlateLocNet.png'
model_save_path = root_directory / 'src' / 'models' / 'PlateLocNet.pth'

def load_data(batch_size=None):
    train_df, val_df, test_df = get_df()

    bbox_config = A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.1,
        clip=True
    )

    train_transform = A.Compose([
        A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=(-8, 8), p=0.3),
        A.Perspective(scale=(0.02, 0.05), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.2),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.GaussNoise(std_range=(10.0 / 255.0, 25.0 / 255.0), p=0.1),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=bbox_config)

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=bbox_config)

    train_ds = LicensePlateObjectDetectionDataset(train_df, 14, transform=train_transform)
    val_ds = LicensePlateObjectDetectionDataset(val_df, 14, transform=val_transform)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    return train_loader, val_loader

def save_plots(history=None):
    epochs = range(len(history['train_loss']))

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].plot(epochs, history['train_loss'], 'r', label='Train loss')
    axs[0, 0].plot(epochs, history['val_loss'], 'b', label='Val loss')
    axs[0, 0].set_title('Total Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(epochs, history['train_box_loss'], 'r', label='Train box loss')
    axs[0, 1].plot(epochs, history['val_box_loss'], 'b', label='Val box loss')
    axs[0, 1].set_title('Box Loss')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(epochs, history['train_true_object_loss'], 'r', label='Train true object loss')
    axs[1, 0].plot(epochs, history['val_true_object_loss'], 'b', label='Val true object loss')
    axs[1, 0].set_title('Objectness Loss')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(epochs, history['train_false_object_loss'], 'r', label='Train false object loss')
    axs[1, 1].plot(epochs, history['val_false_object_loss'], 'b', label='Val false object loss')
    axs[1, 1].set_title('No-Objectness Loss')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(charts_save_path)
    plt.close()
    
def train_model(model=None, train_loader=None, val_loader=None, epochs=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    history = {
        'train_loss': [], 'val_loss': [],
        'train_box_loss': [], 'val_box_loss': [],
        'train_true_object_loss': [], 'val_true_object_loss': [],
        'train_false_object_loss': [], 'val_false_object_loss': []
    }

    model.to(device)

    criterion = get_loss()

    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_train_loss, running_train_box_loss, running_train_true_object_loss, running_train_false_object_loss = 0.0, 0.0, 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            output = model(images)
            train_loss, train_box_loss, train_true_object_loss, train_false_object_loss = criterion(output, targets)
            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item()
            running_train_box_loss += train_box_loss.item()
            running_train_true_object_loss += train_true_object_loss.item()
            running_train_false_object_loss += train_false_object_loss.item()

            pbar.set_postfix({
                'train_loss': f"{train_loss.item():.3f}",
                'train_box_loss': f"{train_box_loss.item():.3f}",
                'train_true_object_loss': f"{train_true_object_loss.item():.3f}",
                'train_false_object_loss': f"{train_false_object_loss.item():.3f}"
            })

        num_train_batches = len(train_loader)
        history['train_loss'].append(running_train_loss / num_train_batches)
        history['train_box_loss'].append(running_train_box_loss / num_train_batches)
        history['train_true_object_loss'].append(running_train_true_object_loss / num_train_batches)
        history['train_false_object_loss'].append(running_train_false_object_loss / num_train_batches)

        model.eval()
        running_val_loss, running_val_box_loss, running_val_true_object_loss, running_val_false_object_loss = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)

                output = model(images)
                val_loss, val_box_loss, val_true_object_loss, val_false_object_loss = criterion(output, targets)

                running_val_loss += val_loss.item()
                running_val_box_loss += val_box_loss.item()
                running_val_true_object_loss += val_true_object_loss.item()
                running_val_false_object_loss += val_false_object_loss.item()

        num_val_batches = len(val_loader)
        history['val_loss'].append(running_val_loss / num_val_batches)
        history['val_box_loss'].append(running_val_box_loss / num_val_batches)
        history['val_true_object_loss'].append(running_val_true_object_loss / num_val_batches)
        history['val_false_object_loss'].append(running_val_false_object_loss / num_val_batches)

        print(f"\n[SUMMARY Epoch {epoch}]")
        print(f"Train -> Train loss: {history['train_loss'][-1]:.4f} | Train box loss: {history['train_box_loss'][-1]:.4f} | Train true object loss: {history['train_true_object_loss'][-1]:.4f} | Train false object loss: {history['train_false_object_loss'][-1]:.4f}")
        print(f"Val   -> Val loss: {history['val_loss'][-1]:.4f} | Val box loss: {history['val_box_loss'][-1]:.4f} | Val true object loss: {history['val_true_object_loss'][-1]:.4f} | Val false object loss: {history['val_false_object_loss'][-1]:.4f}")

        epoch_val_loss = history['val_loss'][-1]

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print("[New best model saved]")

        save_plots(history=history)

if __name__ == "__main__":
    train_loader, val_loader = load_data(batch_size=32)
    model = PlateLocNet()
    train_model(model=model, train_loader=train_loader, val_loader=val_loader, epochs=100)