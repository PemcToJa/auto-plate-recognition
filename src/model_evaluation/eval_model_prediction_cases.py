import numpy as np
import matplotlib.pyplot as plt
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.data_curation.dataframe_assembler import get_df
from src.dataset_construction.dataset import LicensePlateObjectDetectionDataset
from src.model_construction.PlateLocLiteNet_Classifier import PlateLocLiteNet
from src.model_construction.PlateLocNet_Classifier import PlateLocNet

def denormalize_image(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = tensor.cpu().numpy().transpose(1, 2, 0)
    image = (image * std + mean) * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def draw_yolo_box(ax, box, color='red', label=None):
    x_center_fraction, y_center_fraction, width_fraction, height_fraction = box.cpu().numpy()

    width = width_fraction * 224
    height = height_fraction * 224
    x = (x_center_fraction * 224) - (width / 2)
    y = (y_center_fraction * 224) - (height / 2)

    rect = plt.Rectangle((x, y), width, height, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    if label:
        ax.text(x, y - 5, label, color=color, fontsize=10, weight='bold', backgroundcolor='black')

def yolo_to_corners(box):
    x_center_fraction, y_center_fraction, width_fraction, height_fraction = box[0].item(), box[1].item(), box[2].item(), box[3].item()
    x_min = x_center_fraction - (width_fraction / 2.0)
    y_min = y_center_fraction - (height_fraction / 2.0)
    x_max = x_center_fraction + (width_fraction / 2.0)
    y_max = y_center_fraction + (height_fraction / 2.0)
    return [x_min, y_min, x_max, y_max]


def calculate_iou(prediction_box, ground_truth_box):
    prediction_corners = yolo_to_corners(prediction_box)
    ground_truth_corners = yolo_to_corners(ground_truth_box)

    left_border_of_intersection = max(prediction_corners[0], ground_truth_corners[0])
    upper_border_of_intersection = max(prediction_corners[1], ground_truth_corners[1])
    right_border_of_intersection = min(prediction_corners[2], ground_truth_corners[2])
    lower_border_of_intersection = min(prediction_corners[3], ground_truth_corners[3])

    width_of_intersection = max(0.0, right_border_of_intersection - left_border_of_intersection)
    height_of_intersection = max(0.0, lower_border_of_intersection - upper_border_of_intersection)
    surface_area_of_intersection = width_of_intersection * height_of_intersection

    surface_area_prediction_box = (prediction_corners[2] - prediction_corners[0]) * (prediction_corners[3] - prediction_corners[1])
    surface_area_ground_truth_box = (ground_truth_corners[2] - ground_truth_corners[0]) * (ground_truth_corners[3] - ground_truth_corners[1])

    total_area_covered_by_both_boxes = surface_area_prediction_box + surface_area_ground_truth_box - surface_area_of_intersection

    if total_area_covered_by_both_boxes == 0:
        return 0.0
    return surface_area_of_intersection / total_area_covered_by_both_boxes

def plot_and_save_canvas(data_list, title="", save_filename="canvas.png"):
    num_images = len(data_list)
    fig, axes = plt.subplots(2, num_images, figsize=(25, 6))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)

    for idx, data in enumerate(data_list):
        ax_pred = axes[0, idx] if num_images > 1 else axes[0]
        ax_pred.imshow(data['display_img_pred'])
        ax_pred.axis('off')

        if idx == 0:
            ax_pred.text(-20, 112, "Prediction", fontsize=12, fontweight='bold',
                         va='center', ha='right', color='red')

        if data['global_box'] is not None:
            draw_yolo_box(ax_pred, data['global_box'], color='red', label=f"{data['highest_conf']:.1%}")
        else:
            ax_pred.text(10, 20, "No plate", color='yellow', weight='bold', backgroundcolor='black')

        ax_pred.set_title(f"Sample {idx + 1}\nIoU: {data['iou']:.3f}", fontsize=11, fontweight='bold')

        ax_gt = axes[1, idx] if num_images > 1 else axes[1]
        ax_gt.imshow(data['display_img_gt'])
        ax_gt.axis('off')

        if idx == 0:
            ax_gt.text(-20, 112, "Ground truth", fontsize=12, fontweight='bold',
                       va='center', ha='right', color='green')

        if data['global_gt_box'] is not None:
            draw_yolo_box(ax_gt, data['global_gt_box'], color='green', label="True")

    plt.tight_layout()
    plt.savefig(save_filename, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()

def evaluate_and_find_extremes(model, model_name, val_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    all_images = []
    all_targets = []
    with torch.no_grad():
        for images, targets in val_loader:
            all_images.append(images)
            all_targets.append(targets)

    all_images = torch.cat(all_images, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    total_images = all_images.size(0)

    results = []
    for image_idx in range(total_images):
        image_tensor = all_images[image_idx].unsqueeze(0).to(device)
        target = all_targets[image_idx].to(device)

        with torch.no_grad():
            prediction = model(image_tensor).squeeze(0)

        display_image_prediction = denormalize_image(all_images[image_idx])
        display_image_ground_truth = display_image_prediction.copy()

        prediction_confidence = prediction[..., 0]
        cell_idx = torch.argmax(prediction_confidence)
        cell_y = cell_idx // 14
        cell_x = cell_idx % 14

        highest_confidence = prediction_confidence[cell_y, cell_x].item()

        global_box = None
        if highest_confidence > 0.4:
            prediction_box = prediction[cell_y, cell_x, 1:5]
            x_global = (cell_x + prediction_box[0]) / 14.0
            y_global = (cell_y + prediction_box[1]) / 14.0
            global_box = torch.tensor([x_global, y_global, prediction_box[2], prediction_box[3]])

        global_ground_truth_box = None
        true_box_idx = torch.where(target[..., 0] == 1)
        if len(true_box_idx[0]) > 0:
            y, x = true_box_idx[0][0], true_box_idx[1][0]
            ground_truth_box = target[y, x, 1:5]

            x_ground_truth_global = (x + ground_truth_box[0]) / 14.0
            y_ground_truth_global = (y + ground_truth_box[1]) / 14.0
            global_ground_truth_box = torch.tensor([x_ground_truth_global, y_ground_truth_global, ground_truth_box[2], ground_truth_box[3]])

        iou_score = 0.0
        if global_box is not None and global_ground_truth_box is not None:
            iou_score = calculate_iou(global_box, global_ground_truth_box)

        results.append({
            'display_img_pred': display_image_prediction,
            'display_img_gt': display_image_ground_truth,
            'global_box': global_box,
            'global_gt_box': global_ground_truth_box,
            'highest_conf': highest_confidence,
            'iou': iou_score
        })

    results_sorted = sorted(results, key=lambda x: x['iou'])

    worst_10 = results_sorted[:10]
    top_10 = results_sorted[-10:][::-1]

    plot_and_save_canvas(top_10, title=f"Top-10 Predictions {model_name}", save_filename=f"top_10_predictions_{model_name}.png")
    plot_and_save_canvas(worst_10, title=f"Worst-10 Predictions {model_name}", save_filename=f"worst_10_predictions_{model_name}.png")

def load_data(batch_size=None):
    _, _, test_df = get_df()

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
        num_workers=4,
        drop_last=False
    )
    return test_loader

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = load_data(batch_size=32)

    models_to_compare = {}
    model1 = PlateLocNet()
    model1.load_state_dict(torch.load('../models/PlateLocNet.pth', map_location=device))
    models_to_compare['PlateLocNet'] = model1

    model2 = PlateLocLiteNet()
    model2.load_state_dict(torch.load('../models/PlateLocLiteNet.pth', map_location=device))
    models_to_compare['PlateLocLiteNet'] = model2

    for model_name, model in models_to_compare.items():
        evaluate_and_find_extremes(model, model_name, test_loader)