# AI Handwriting Comparison (Minor Project)

An end‑to‑end **Streamlit** app that compares two handwriting images and predicts if they are from the **same writer** or **different writers**.

## 🚀 Features
- Upload two images (JPG/PNG).
- Robust preprocessing: grayscale, binarize, denoise, deskew.
- Feature extraction: **HOG**, **LBP**, and simple stroke/texture stats.
- Similarity with **cosine distance** + adjustable threshold.
- Visual feedback: similarity score, decision, and intermediate previews.
- Optional: toggle a lightweight CNN embedding (if PyTorch installed) as an add‑on.

> This is deliberately lightweight (no big pretrained weights) so it runs easily on typical student laptops.
