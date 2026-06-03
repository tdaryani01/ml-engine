import numpy as np
import pandas as pd

def generate_hard_regression(n_samples=1000):
    # X from 0 to 10
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    # The function: sin(x) * sin(x^2)
    # This creates a signal that gets 'crunched' as X increases
    y = np.sin(X) * np.sin(X**2) + np.random.normal(0, 0.02, size=X.shape)
    
    df = pd.DataFrame(np.hstack((X, y)), columns=["Velocity", "Outcome"])
    df.to_csv(".\\data\\generator\\hard_regression.csv", index=False)
    print("Hard regression data generated: sinc(x) * sin(x^2)")

generate_hard_regression()