import numpy as np
import pandas as pd
from sklearn.datasets import make_moons

class NestedMoonsGenerator:
    def __init__(self, n_samples=1000, noise=0.15, random_state=42):
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state

    def generate(self):
        # Generate interlocking moons
        X, y = make_moons(n_samples=self.n_samples, noise=self.noise, random_state=self.random_state)
        
        # Combine into a single matrix
        dataset = np.hstack((X, y.reshape(-1, 1)))
        
        # Shuffle
        np.random.shuffle(dataset)
        return dataset

    def to_csv(self, filename="moons_data.csv"):
        dataset = self.generate()
        df = pd.DataFrame(dataset, columns=["Velocity", "Movement", "Outcome"])
        df.to_csv(filename, index=False)
        print(f"Generated {self.n_samples} samples. Saved to {filename}")
        return df

# Usage
generator = NestedMoonsGenerator(n_samples=1000, noise=0.15)
generator.to_csv(".\data\generator\moons_data.csv")