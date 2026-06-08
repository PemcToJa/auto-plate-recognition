import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data_curation.dataframe_assembler import get_df
from src.dataset_construction.dataset import LicensePlateObjectDetectionDataset
from src.model_construction.PlateLocLiteNet_Classifier import PlateLocLiteNet
from src.model_construction.PlateLocNet_Classifier import PlateLocNet

def load_data(batch_size):
    train_df, val_df, test_df = get_df()

    bbox_config = A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.1,
        clip=True
    )

    test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=bbox_config)

    test_ds = LicensePlateObjectDetectionDataset(test_df, 14, transform=test_transform)

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    return test_loader

def get_predictions(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)

            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_predictions, all_targets

def convert_yolo_to_xyxy_pixels(cell_column, cell_row, box):
    x_position_inside_cell = box[0].item()
    y_position_inside_cell = box[1].item()
    width_fraction = box[2].item()
    height_fraction = box[3].item()

    x_center_fraction = (cell_column + x_position_inside_cell) / 14.0
    y_center_fraction = (cell_row + y_position_inside_cell) / 14.0

    width = width_fraction * 224
    height = height_fraction * 224
    x_center = x_center_fraction * 224
    y_center = y_center_fraction * 224

    x_min = x_center - (width / 2.0)
    y_min = y_center - (height / 2.0)
    x_max = x_center + (width / 2.0)
    y_max = y_center + (height / 2.0)

    return [x_min, y_min, x_max, y_max]

def calculate_iou(prediction_box, ground_truth_box):
    left_border_of_intersection = max(prediction_box[0], ground_truth_box[0])
    upper_border_of_intersection = max(prediction_box[1], ground_truth_box[1])
    right_border_of_intersection = min(prediction_box[2], ground_truth_box[2])
    lower_border_of_intersection = min(prediction_box[3], ground_truth_box[3])

    width_of_intersection = max(0, right_border_of_intersection - left_border_of_intersection)
    height_of_intersection = max(0, lower_border_of_intersection - upper_border_of_intersection)
    surface_area_of_intersection = width_of_intersection * height_of_intersection

    surface_area_prediction_box = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
    surface_area_ground_truth_box = (ground_truth_box[2] - ground_truth_box[0]) * (ground_truth_box[3] - ground_truth_box[1])
    total_area_covered_by_both_boxes = surface_area_prediction_box + surface_area_ground_truth_box - surface_area_of_intersection

    if total_area_covered_by_both_boxes == 0:
        return 0.0
    return surface_area_of_intersection / total_area_covered_by_both_boxes

def evaluate_models_global_iou(models_dict):
    test_loader = load_data(batch_size=32)

    model_names = []
    mean_iou_s = []

    for model_name, model in models_dict.items():
        all_predictions, all_targets = get_predictions(model, test_loader)
        images_amount = all_predictions.shape[0]

        iou_scores = []

        for single_image in range(images_amount):
            predicted_image = all_predictions[single_image]
            target_image = all_targets[single_image]

            true_box_idx = torch.where(target_image[..., 0] == 1)

            predicted_confidence = predicted_image[..., 0]
            cell_idx = torch.argmax(predicted_confidence)
            prediction_y_cell = int(cell_idx // 14)
            prediction_x_cell = int(cell_idx % 14)

            ground_truth_y_cell, ground_truth_x_cell = true_box_idx[0][0].item(), true_box_idx[1][0].item()
            ground_truth_box = target_image[ground_truth_y_cell, ground_truth_x_cell, 1:5]
            prediction_box = predicted_image[prediction_y_cell, prediction_x_cell, 1:5]

            ground_truth_box_object = convert_yolo_to_xyxy_pixels(ground_truth_x_cell, ground_truth_y_cell, ground_truth_box)
            prediction_box_object = convert_yolo_to_xyxy_pixels(prediction_x_cell, prediction_y_cell, prediction_box)

            current_iou = calculate_iou(prediction_box_object, ground_truth_box_object)
            iou_scores.append(current_iou)

        mean_iou = np.mean(iou_scores)
        model_names.append(model_name)
        mean_iou_s.append(mean_iou)

    plt.figure(figsize=(max(5, len(model_names) * 1.5), 6))

    bars = plt.bar(model_names, mean_iou_s, color='dodgerblue', width=0.4, edgecolor='black', linewidth=1.2)

    plt.ylabel('Metric Value (IoU Score)', fontsize=12, fontweight='bold')
    plt.title('Models Localization Accuracy Comparison', fontsize=12,
              fontweight='bold', pad=15)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f'{height * 100:.2f}%',
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    plt.tight_layout()
    plt.savefig('global_mean_iou_metric.png', bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_to_compare = {}
    model1 = PlateLocNet()
    model1.load_state_dict(torch.load('../models/PlateLocNet.pth', map_location=device))
    models_to_compare['PlateLocNet'] = model1

    model2 = PlateLocLiteNet()
    model2.load_state_dict(torch.load('../models/PlateLocLiteNet.pth', map_location=device))
    models_to_compare['PlateLocLiteNet'] = model2

    evaluate_models_global_iou(models_to_compare)