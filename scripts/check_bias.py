import numpy as np

archive = np.load("deployed_model.npz")
b_out = archive["b_2"].ravel()  # Flatten array to 1D

print("\n" + "=" * 50)
print("OUTPUT LAYER BIASES (b_2):")
print(f"  Class 0 (Clean)      : {float(b_out[0]):.6f}")
print(f"  Class 1 (Suspicious) : {float(b_out[1]):.6f}")
print(f"  Class 2 (Malicious)  : {float(b_out[2]):.6f}")
print("=" * 50 + "\n")