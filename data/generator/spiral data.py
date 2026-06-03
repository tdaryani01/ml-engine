import numpy as np
import pandas as pd

def generate_spiral_data(samples_per_class=343):
    X = np.zeros((samples_per_class * 2, 2))
    y = np.zeros((samples_per_class * 2, 1))
    
    for class_number in range(2):
        ix = range(samples_per_class * class_number, samples_per_class * (class_number + 1))
        # Radius
        r = np.linspace(0.0, 1, samples_per_class)
        # Angle (Theta)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, samples_per_class) + np.random.randn(samples_per_class) * 0.2
        
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = class_number
        
    return X, y

# Generate the data
X_spiral, y_spiral = generate_spiral_data()

# Scale features to match your previous distributions (roughly -7 to +7)
X_spiral = X_spiral * 7.0 

# Create DataFrame and shuffle
dataset = np.hstack((X_spiral, y_spiral))
np.random.shuffle(dataset)
df = pd.DataFrame(dataset, columns=["Velocity", "Movement", "Outcome"])

df.to_csv(".\data\generator\spiral_data.csv", index=False)
print(f"Generated {len(df)} rows. Saved to spiral_data.csv")