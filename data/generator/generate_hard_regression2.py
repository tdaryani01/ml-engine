# data/generator/chaotic_oscillator.py
import numpy as np
import pandas as pd
import os

def generate_chaotic_regression_data(num_samples=6000, noise_std=0.03, out_dir="data/generator"):
    """
    Generates a highly non-linear, continuous 3-feature regression dataset.
    Replaces the step discontinuity with a steep, smooth anharmonic transition well.
    """
    np.random.seed(101)
    
    Time = np.random.uniform(0.0, 4.0, size=(num_samples, 1))
    Radius = np.random.uniform(0.5, 2.5, size=(num_samples, 1))
    Angle = np.random.uniform(0.0, 2 * np.pi, size=(num_samples, 1))
    
    # High-frequency spatial wave mechanics
    frequency_term = np.exp(Time)
    macro_wave = np.sin(frequency_term * Radius) 
    spatial_coupling = np.cos(Angle * 3.0) / (Radius + 0.1)
    
    # =====================================================================
    # CRITICAL FIX: Replace step-barrier with a steep, smooth tanh transition
    # =====================================================================
    # Sharpness scale (20.0) creates a steep cliff, but keeps the manifold continuous
    coordinate_intersection = (Time * Radius) - 4.5
    smooth_cliff = 1.0 + np.tanh(coordinate_intersection * 20.0) # Smooth transition between 0 and 2
    
    # Construct continuous target surface
    y_clean = (macro_wave * spatial_coupling) + smooth_cliff
    
    # Inject Gaussian Noise
    noise = np.random.normal(0, noise_std, size=y_clean.shape)
    y_noisy = y_clean + noise
    
    df = pd.DataFrame({
        "Time": Time.ravel(),
        "Radius": Radius.ravel(),
        "Angle": Angle.ravel(),
        "Outcome": y_noisy.ravel()
    })
    
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, "chaotic_oscillator.csv")
    df.to_csv(file_path, index=False)
    print(f"[Data Generator] Continuous manifold generated successfully at: {file_path}")

if __name__ == "__main__":
    generate_chaotic_regression_data()