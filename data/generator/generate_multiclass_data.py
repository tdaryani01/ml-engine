# generate_multiclass_data.py
import os
import numpy as np

def generate_spiral_dataset(file_path="data/generator/robotic_multiclass_data.csv", samples_per_class=2000, noise=0.2):
    """
    Manufactures an interwoven 3-armed spiral coordinate matrix 
    and exports it to a standard CSV.
    """
    print(f"Manufacturing 3-class spiral dataset ({samples_per_class * 3} total samples)...")
    
    num_classes = 3
    X = np.zeros((samples_per_class * num_classes, 2))
    y = np.zeros((samples_per_class * num_classes, 1), dtype=int)
    
    np.random.seed(42)  # Fixed seed for reproducible test patterns
    
    for class_idx in range(num_classes):
        # Calculate array slice indices
        ix = range(samples_per_class * class_idx, samples_per_class * (class_idx + 1))
        
        # Radii profiles
        r = np.linspace(0.0, 10.0, samples_per_class)
        
        # Angular space offset by 2*pi/3 for each arm, tracking with stochastic noise
        theta = (np.linspace(class_idx * 2.5, (class_idx + 2.5) * 2.5, samples_per_class) 
                 + np.random.randn(samples_per_class) * noise)
        
        # Assign spatial coordinates (Velocity, Movement)
        X[ix] = np.c_[r * np.sin(theta), r * np.cos(theta)]
        y[ix] = class_idx
        
    # Stack features and categorical integer markers horizontally
    dataset_matrix = np.hstack([X, y])
    
    # Ensure directory framework exists
    os.makedirs(os.path.dirname(os.path.abspath(file_path)) if os.path.dirname(file_path) else '.', exist_ok=True)
    
    # Save target coordinates directly to disk
    header = "Velocity,Movement,Target_Class"
    np.savetxt(file_path, dataset_matrix, delimiter=',', header=header, comments='')
    print(f"Successfully generated and committed multi-class layout to: {file_path}")

if __name__ == "__main__":
    # Fire the generation sequence once on demand
    generate_spiral_dataset()