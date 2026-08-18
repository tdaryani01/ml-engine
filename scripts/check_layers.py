import numpy as np

archive = np.load("models/deployed_model.npz")

w0, b0 = archive["w_0"], archive["b_0"].ravel()
w1, b1 = archive["w_1"], archive["b_1"].ravel()
w2, b2 = archive["w_2"], archive["b_2"].ravel()

def forward_debug(x):
    z0 = np.dot(x, w0) + b0
    a0 = np.maximum(0, z0)
    
    z1 = np.dot(a0, w1) + b1
    a1 = np.maximum(0, z1)
    
    logits = np.dot(a1, w2) + b2
    
    # Softmax
    shift_logits = logits - np.max(logits)
    exps = np.exp(shift_logits)
    probs = exps / np.sum(exps)
    return logits, probs

# Test zero vector
l_zero, p_zero = forward_debug(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
l_live, p_live = forward_debug(np.array([0.0392, 0.0, 0.0, 0.0, 0.0]))

print("\n" + "=" * 60)
print("RAW LOGITS (Before Softmax):")
print(f"  Zero Vector Logits [Clean, Susp, Mal] : {l_zero.round(4)}")
print(f"  Live Vector Logits [Clean, Susp, Mal] : {l_live.round(4)}")
print("=" * 60 + "\n")