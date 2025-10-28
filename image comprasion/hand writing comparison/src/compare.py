import numpy as np

def cosine_similarity(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)

def decide(similarity, threshold=0.5):
    # similarity in [0..1]; higher means more similar
    same = similarity >= threshold
    return same, ("Same Writer ✅" if same else "Different Writer ❌")
