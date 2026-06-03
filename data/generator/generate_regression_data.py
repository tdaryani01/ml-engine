# generate_regression_data.py
import numpy as np

def generate_synthetic_telemetry(file_path="robotic_regression_data.csv", num_samples=1200, noise_factor=0.1):
    print(f"Manufacturing continuous regression dataset ({num_samples} samples)...")
    
    # Generate random continuous input features (e.g., Velocity and Movement states)
    np.random.seed(42)  # Fixed seed for reproducible test scenarios
    velocity = np.random.uniform(-5.0, 5.0, (num_samples, 1))
    movement = np.random.uniform(-5.0, 5.0, (num_samples, 1))
    
    # Define a complex, non-linear continuous target function to challenge the network
    # Target Angle = sin(velocity) * cos(movement) + polynomial interaction + noise
    target_angle = (np.sin(velocity) * np.cos(movement)) + (0.05 * (velocity ** 2)) + np.random.normal(0, noise_factor, (num_samples, 1))
    
    # Stack features and continuous targets horizontally
    dataset_matrix = np.hstack([velocity, movement, target_angle])
    
    # Save directly to a standard CSV matching your loader layout
    header = "Velocity,Movement,Target_Angle"
    np.savetxt(file_path, dataset_matrix, delimiter=',', header=header, comments='')
    print(f"Successfully generated and saved static asset to: {file_path}")

if __name__ == "__main__":
    # Run this once on demand whenever you want a fresh regression testing file
    generate_synthetic_telemetry(noise_factor=0.0)