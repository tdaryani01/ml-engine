import os
import numpy as np
import pandas as pd

def generate_double_helix_swiss_roll(n_samples=2000, noise=0.1):
    np.random.seed(42)
    n = n_samples // 2
    
    # Class 0: First helix
    t0 = np.linspace(1.5 * np.pi, 4.5 * np.pi, n)
    x0 = t0 * np.cos(t0)
    y0 = np.linspace(0, 10, n)
    z0 = t0 * np.sin(t0)
    
    # Class 1: Second helix (180 degrees out of phase)
    t1 = np.linspace(1.5 * np.pi, 4.5 * np.pi, n)
    x1 = t1 * np.cos(t1 + np.pi)
    y1 = np.linspace(0, 10, n)
    z1 = t1 * np.sin(t1 + np.pi)
    
    X0 = np.vstack((x0, y0, z0)).T + np.random.randn(n, 3) * noise
    X1 = np.vstack((x1, y1, z1)).T + np.random.randn(n, 3) * noise
    
    df0 = pd.DataFrame(X0, columns=['X', 'Y', 'Z'])
    df0['Target'] = 0
    
    df1 = pd.DataFrame(X1, columns=['X', 'Y', 'Z'])
    df1['Target'] = 1
    
    df = pd.concat([df0, df1], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs("./data/generator", exist_ok=True)
    df.to_csv("./data/generator/swiss_helix.csv", index=False)
    print("Dataset generated successfully at ./data/generator/swiss_helix.csv")

if __name__ == "__main__":
    generate_double_helix_swiss_roll()