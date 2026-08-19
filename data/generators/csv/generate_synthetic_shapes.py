# data/generators/generate_synthetic_shapes.py
import os
import csv
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def draw_circle(grid, c_y, c_x, radius, channels=3):
    H, W = grid.shape[1], grid.shape[2]
    y, x = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((x - c_x)**2 + (y - c_y)**2)
    mask = dist_from_center <= radius
    for c in range(channels):
        grid[c, mask] = 1.0


def draw_square(grid, c_y, c_x, half_size, channels=3):
    H, W = grid.shape[1], grid.shape[2]
    y_min, y_max = max(0, c_y - half_size), min(H, c_y + half_size + 1)
    x_min, x_max = max(0, c_x - half_size), min(W, c_x + half_size + 1)
    for c in range(channels):
        grid[c, y_min:y_max, x_min:x_max] = 1.0


def draw_cross(grid, c_y, c_x, arm_len, thickness=1, channels=3):
    H, W = grid.shape[1], grid.shape[2]
    # Horizontal arm
    x_min, x_max = max(0, c_x - arm_len), min(W, c_x + arm_len + 1)
    y_min, y_max = max(0, c_y - thickness), min(H, c_y + thickness + 1)
    for c in range(channels):
        grid[c, y_min:y_max, x_min:x_max] = 1.0

    # Vertical arm
    x_min, x_max = max(0, c_x - thickness), min(W, c_x + thickness + 1)
    y_min, y_max = max(0, c_y - arm_len), min(H, c_y + arm_len + 1)
    for c in range(channels):
        grid[c, y_min:y_max, x_min:x_max] = 1.0


def draw_diamond(grid, c_y, c_x, radius, channels=3):
    H, W = grid.shape[1], grid.shape[2]
    y, x = np.ogrid[:H, :W]
    manhattan_dist = np.abs(x - c_x) + np.abs(y - c_y)
    mask = manhattan_dist <= radius
    for c in range(channels):
        grid[c, mask] = 1.0


def generate_shapes_dataset(output_csv_path: str, num_samples_per_class: int = 250,
                            channels: int = 3, height: int = 28, width: int = 28,
                            noise_level: float = 0.15):
    """
    Generates synthetic multi-channel spatial patterns:
      Class 0: Circle
      Class 1: Square
      Class 2: Cross
      Class 3: Diamond
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    num_classes = 4
    total_samples = num_samples_per_class * num_classes
    feature_dim = channels * height * width

    logging.info(f"[Shape Generator] Synthesizing {total_samples} samples across {num_classes} classes...")
    logging.info(f"[Shape Generator] Tensor Shape: ({channels}, {height}, {width}) -> Flattened Feature Dim: {feature_dim}")

    records = []
    labels = []

    for class_idx in range(num_classes):
        for _ in range(num_samples_per_class):
            img = np.zeros((channels, height, width), dtype=np.float32)

            # Randomize center coordinates and scales
            c_y = np.random.randint(height // 4, height - height // 4)
            c_x = np.random.randint(width // 4, width - width // 4)
            scale = np.random.randint(4, min(height, width) // 3)

            if class_idx == 0:
                draw_circle(img, c_y, c_x, radius=scale, channels=channels)
            elif class_idx == 1:
                draw_square(img, c_y, c_x, half_size=scale, channels=channels)
            elif class_idx == 2:
                draw_cross(img, c_y, c_x, arm_len=scale, thickness=1, channels=channels)
            elif class_idx == 3:
                draw_diamond(img, c_y, c_x, radius=scale, channels=channels)

            # Inject Gaussian sensor noise and clip
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, img.shape)
                img = np.clip(img + noise, 0.0, 1.0)

            records.append(img.flatten())
            labels.append(class_idx)

    # Shuffle instances
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    header = [f"px_{i}" for i in range(feature_dim)] + ["target"]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx in indices:
            row = np.round(records[idx], 4).tolist() + [labels[idx]]
            writer.writerow(row)

    logging.info(f"[Shape Generator] Successfully saved {total_samples} samples to: {output_csv_path}")


if __name__ == "__main__":
    target_path = os.path.join("data", "samples", "csv", "synthetic_shapes.csv")
    generate_shapes_dataset(target_path, num_samples_per_class=300, channels=3, height=28, width=28)