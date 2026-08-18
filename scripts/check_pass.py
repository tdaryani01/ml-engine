import numpy as np
from core.inference import ProxyShieldInferenceEngine

engine = ProxyShieldInferenceEngine("deployed_model.npz")

# Test 1: Literal Zero Vector
zero_vec = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
p_zero = engine.forward_pass(zero_vec)[0]

# Test 2: Your Live Prompt Vector
live_vec = np.array([[0.0392, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
p_live = engine.forward_pass(live_vec)[0]

print("\n" + "=" * 60)
print("TEST 1: ALL-ZERO VECTOR [0, 0, 0, 0, 0]")
print(f"  Clean (Class 0)      : {p_zero[0]:.6f}")
print(f"  Suspicious (Class 1) : {p_zero[1]:.6f}")
print(f"  Malicious (Class 2)  : {p_zero[2]:.6f}")

print("-" * 60)
print("TEST 2: LIVE VECTOR [0.0392, 0, 0, 0, 0]")
print(f"  Clean (Class 0)      : {p_live[0]:.6f}")
print(f"  Suspicious (Class 1) : {p_live[1]:.6f}")
print(f"  Malicious (Class 2)  : {p_live[2]:.6f}")
print("=" * 60 + "\n")