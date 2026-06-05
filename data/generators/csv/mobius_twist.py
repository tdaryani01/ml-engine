import os
import numpy as np
import pandas as pd

def generate_twin_mobius(n_samples=2000, noise=0.05):
    np.random.seed(42)
    n = n_samples // 2
    
    # Common parameters
    length = np.linspace(0, 2 * np.pi, n)
    width = np.linspace(-0.5, 0.5, n)
    
    # Class 0: First Ribbon
    w0 = width
    x0 = (1 + w0 * np.cos(length / 2)) * np.cos(length)
    y0 = (1 + w0 * np.cos(length / 2)) * np.sin(length)
    z0 = w0 * np.sin(length / 2)
    
    # Class 1: Interlocking Parallel Ribbon (shifted in width and phase)
    w1 = width + 0.2
    phase_shift = length + np.pi
    x1 = (1 + w1 * np.cos(phase_shift / 2)) * np.cos(phase_shift)
    y1 = (1 + w1 * np.cos(phase_shift / 2)) * np.sin(phase_shift)
    z1 = w1 * np.sin(phase_shift / 2)
    
    X0 = np.vstack((x0, y0, z0)).T + np.random.randn(n, 3) * noise
    X1 = np.vstack((x1, y1, z1)).T + np.random.randn(n, 3) * noise
    
    df0 = pd.DataFrame(X0, columns=['X', 'Y', 'Z'])
    df0['Target'] = 0
    
    df1 = pd.DataFrame(X1, columns=['X', 'Y', 'Z'])
    df1['Target'] = 1
    
    df = pd.concat([df0, df1], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs("./data/generator", exist_ok=True)
    df.to_csv("./data/generator/mobius_twist.csv", index=False)
    print("Dataset generated at ./data/generator/mobius_twist.csv")

if __name__ == "__main__":
    generate_twin_mobius()