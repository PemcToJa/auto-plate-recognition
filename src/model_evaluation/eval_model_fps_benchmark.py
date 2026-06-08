import time
import matplotlib.pyplot as plt
import torch

from src.model_construction.PlateLocLiteNet_Classifier import PlateLocLiteNet
from src.model_construction.PlateLocNet_Classifier import PlateLocNet

def benchmark_models_speed(models_dict, num_warmup=50, num_iters=300):
    device = "cpu"

    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    model_names = []
    fps_results = []
    latency_results = []

    for model_name, model in models_dict.items():
        model.to(device)
        model.eval()

        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(num_iters):
                _ = model(dummy_input)

            if device == "cuda":
                torch.cuda.synchronize()

        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_latency_ms = (total_time / num_iters) * 1000.0
        fps = num_iters / total_time

        model_names.append(model_name)
        fps_results.append(fps)
        latency_results.append(avg_latency_ms)

    plt.figure(figsize=(max(5, len(model_names) * 1.5), 6))

    bars = plt.bar(model_names, fps_results, color='mediumseagreen', width=0.4, edgecolor='black', linewidth=1.2)

    plt.ylabel('Frames Per Second (FPS)', fontsize=12, fontweight='bold')
    plt.title('Inference Speed Comparison', fontsize=12, fontweight='bold', pad=15)

    plt.ylim(0, max(fps_results) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='bold')

    for bar, latency in zip(bars, latency_results):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + (max(fps_results) * 0.02),
                 f'{height:.1f} FPS\n({latency:.2f} ms)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

    plt.tight_layout()
    plt.savefig('model_speed_fps_benchmark.png', bbox_inches='tight', dpi=150)
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

    benchmark_models_speed(models_to_compare, num_warmup=50, num_iters=500)