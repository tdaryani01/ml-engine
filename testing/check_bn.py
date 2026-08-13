import numpy as np

archive = np.load("models/deployed_model.npz")

print("\n" + "=" * 60)
print("BATCH NORM RUNNING STATISTICS IN DEPLOYED MODEL:")
print("-" * 60)

num_bn_layers = len([k for k in archive.keys() if k.startswith("rmean_")])
for i in range(num_bn_layers):
    rmean = archive[f"rmean_{i}"]
    rvar = archive[f"rvar_{i}"]
    gamma = archive[f"gamma_{i}"]
    beta = archive[f"beta_{i}"]
    
    print(f"Layer {i} BatchNorm Params:")
    print(f"  running_mean (shape {rmean.shape}) -> min: {np.min(rmean):.4f}, max: {np.max(rmean):.4f}, mean: {np.mean(rmean):.4f}")
    print(f"  running_var  (shape {rvar.shape}) -> min: {np.min(rvar):.4f}, max: {np.max(rvar):.4f}, mean: {np.mean(rvar):.4f}")
    print(f"  gamma        (shape {gamma.shape}) -> min: {np.min(gamma):.4f}, max: {np.max(gamma):.4f}")
    print(f"  beta         (shape {beta.shape}) -> min: {np.min(beta):.4f}, max: {np.max(beta):.4f}")
    print("-" * 60)

print("=" * 60 + "\n")