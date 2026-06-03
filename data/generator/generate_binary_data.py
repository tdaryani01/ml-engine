import numpy as np
import pandas as pd

class BinaryDataGenerator:
    def __init__(self, n_samples=1000, n_features=2, separation=4.0, noise=1.0, random_state=42):
        """
        Args:
            n_samples (int): Total number of rows to generate.
            n_features (int): Number of input columns (X).
            separation (float): The distance between the centers of the two classes.
                                High separation = easy to learn.
            noise (float): The spread/variance of the data points around their center.
                           High noise = overlapping classes.
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.separation = separation
        self.noise = noise
        np.random.seed(random_state)

    def generate(self):
        # Split samples evenly between class 0 and class 1
        n_class_0 = self.n_samples // 2
        n_class_1 = self.n_samples - n_class_0

        # Define distinct centers for the two classes
        center_0 = np.ones(self.n_features) * (-self.separation / 2)
        center_1 = np.ones(self.n_features) * (self.separation / 2)

        # Generate normally distributed points around the centers
        X_0 = center_0 + np.random.randn(n_class_0, self.n_features) * self.noise
        X_1 = center_1 + np.random.randn(n_class_1, self.n_features) * self.noise

        # Assign targets
        y_0 = np.zeros((n_class_0, 1))
        y_1 = np.ones((n_class_1, 1))

        # Combine the data
        X = np.vstack((X_0, X_1))
        y = np.vstack((y_0, y_1))
        dataset = np.hstack((X, y))

        # Shuffle the dataset so batches get a mix of both classes
        np.random.shuffle(dataset)

        return dataset

    def to_csv(self, filename="synthetic_binary_data.csv"):
        dataset = self.generate()
        
        # Create dynamic column names (Feature_1, Feature_2, ..., Target)
        columns = [f"Feature_{i+1}" for i in range(self.n_features)] + ["Target"]
        
        df = pd.DataFrame(dataset, columns=columns)
        df.to_csv(filename, index=False)
        
        print(f"Successfully generated {self.n_samples} samples with {self.n_features} features.")
        print(f"Saved to: {filename}")
        print(f"Target distribution:\n{df['Target'].value_counts().to_string()}")
        
        return df

# --- How to use it ---
if __name__ == "__main__":
    # 1. Create highly separated data (Easy Mode)
    # generator = BinaryDataGenerator(n_samples=2000, n_features=2, separation=5.0, noise=0.5)
    # df = generator.to_csv(".\data\generator\easy_binary_data.csv")
    
    # 2. To test later: Create overlapping data (Hard Mode)
    hard_generator = BinaryDataGenerator(n_samples=2000, separation=1.0, noise=2.0)
    hard_generator.to_csv(".\data\generator\hard_binary_data.csv")