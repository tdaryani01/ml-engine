import os
import csv
import urllib.request

def download_banknote_benchmark(output_path=r".\data\generator\robotic_regression_data.csv"):
    """
    Downloads the real-world Banknote Authentication dataset and parses the 
    top two numerical features to fit the Velocity/Movement framework pipeline.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    print("[Pipeline] Fetching real-world banknote currency matrix data...")
    with urllib.request.urlopen(url) as response:
        raw_data = response.read().decode('utf-8').strip().split('\n')

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Keeps headers identical to your production yaml expectations
        writer.writerow(["Velocity", "Movement", "Target"])
        
        for line in raw_data:
            if not line: continue
            parts = line.split(',')
            # Extract Feature 0 (Variance) and Feature 1 (Skewness) as your inputs
            v, m, target = float(parts[0]), float(parts[1]), int(parts[4])
            writer.writerow([v, m, target])
            
    print(f"[Pipeline] Successfully populated {len(raw_data)} real-world vector samples.")

download_banknote_benchmark()