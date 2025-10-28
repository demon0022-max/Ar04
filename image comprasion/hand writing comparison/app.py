import streamlit as st
import numpy as np
from PIL import Image
import cv2

from src.features import preprocess_and_featurize
from src.compare import cosine_similarity, decide

st.set_page_config(page_title="AI Handwriting Comparator", page_icon="✍️", layout="wide")
st.title("✍️ AI Handwriting Comparison")
st.caption("Educational demo — compares two handwriting images and predicts Same/Different writer")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Similarity threshold", 0.30, 0.80, 0.50, 0.01)
    show_intermediate = st.checkbox("Show preprocessing preview", True)
    use_cnn = st.checkbox("Use optional tiny CNN embedding (advanced)", False)
    st.markdown("> Tip: Keep threshold near 0.50 and tweak based on your dataset.")

col1, col2 = st.columns(2)
with col1:
    img1_file = st.file_uploader("Upload first handwriting image", type=["png","jpg","jpeg"], key="img1")
with col2:
    img2_file = st.file_uploader("Upload second handwriting image", type=["png","jpg","jpeg"], key="img2")

def load_cv2(file):
    image = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def to_rgb(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def ensure_uint8(img):
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

btn = st.button("🔍 Compare", use_container_width=True, type="primary")

# Optional CNN
cnn_available = False
if use_cnn:
    try:
        from src.tiny_cnn import extract_embedding
        cnn_available = True
    except Exception as e:
        st.warning(f"Tiny CNN not available: {e}")
        cnn_available = False

if btn:
    if not img1_file or not img2_file:
        st.error("Please upload both images.")
    else:
        bgr1 = load_cv2(img1_file)
        bgr2 = load_cv2(img2_file)

        proc1, feat1, meta1 = preprocess_and_featurize(bgr1)
        proc2, feat2, meta2 = preprocess_and_featurize(bgr2)

        if cnn_available:
            try:
                emb1 = extract_embedding(proc1)
                emb2 = extract_embedding(proc2)
                # Concatenate classical + CNN embeddings
                feat1 = np.concatenate([feat1, emb1]).astype(np.float32)
                feat2 = np.concatenate([feat2, emb2]).astype(np.float32)
                # L2 normalize concatenated features
                for v in (feat1, feat2):
                    n = np.linalg.norm(v) + 1e-8
                    v /= n
            except Exception as e:
                st.warning(f"Failed to use CNN embeddings, falling back to classical: {e}")

        sim = cosine_similarity(feat1, feat2)
        same, msg = decide(sim, threshold)

        st.subheader("Result")
        st.metric("Similarity (0..1)", f"{sim:.3f}")
        st.success(msg) if same else st.error(msg)

        if show_intermediate:
            st.subheader("Preprocessing Preview")
            c1, c2 = st.columns(2)
            with c1:
                st.image(to_rgb(proc1), caption=f"Image 1 (deskew={meta1['deskew_angle']:.2f}°)")
            with c2:
                st.image(to_rgb(proc2), caption=f"Image 2 (deskew={meta2['deskew_angle']:.2f}°)")

st.divider()
st.markdown("### Try demo images")
c1, c2, c3 = st.columns(3)
c1.image("sample_data/sample_a_1.png", caption="sample_a_1.png")
c2.image("sample_data/sample_a_2.png", caption="sample_a_2.png")
c3.image("sample_data/sample_b_1.png", caption="sample_b_1.png")
