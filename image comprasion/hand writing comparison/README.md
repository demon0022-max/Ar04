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

---

## 🧩 Project Structure
```
handwriting_comparator/
│
├── app.py                 # Streamlit UI
├── requirements.txt
├── README.md
└── src/
    ├── features.py        # Preprocess + features (HOG, LBP, stats)
    ├── compare.py         # Similarity + decision logic
    └── tiny_cnn.py        # Optional mini-CNN embedding (off by default)
└── sample_data/
    ├── sample_a_1.png
    ├── sample_a_2.png     # same "writer" as a_1 (synthetic)
    └── sample_b_1.png     # different writer (synthetic)
```

---

## 📦 Installation
Create and activate a virtual environment, then install requirements.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

If you face OpenCV GUI issues on servers, we use **opencv-python-headless**.

---

## ▶️ Run the app
```bash
streamlit run app.py
```
This starts a local server and opens the app in your browser.

---

## 🖼️ Using the app
1. Upload **two** handwriting images (cropped word/line/page all work).
2. Optionally adjust threshold (default tuned ≈ 0.45–0.55 for our feature mix).
3. Click **Compare** to see:
   - Similarity score (0 to 1 where 1 is identical).
   - Decision: **Same writer** or **Different writer**.
   - Preprocessing previews of each image.

> Tip: For best results, **crop** to the same type of content (e.g., both lines or both words) and ensure good contrast.

---

## 📐 How it works
- **Preprocessing**
  - grayscale → median denoise → adaptive binarize
  - **deskew** via Hough-based angle estimation
  - resize to 256×256, center pad
- **Features**
  - HOG (oriented gradients texture)
  - LBP histogram (micro‑texture)
  - Simple stats (ink ratio, stroke density, edge density, contour complexity)
- **Compare**
  - Normalize vectors → **cosine similarity**
  - Decision by threshold (user‑adjustable)

---

## 🧪 Demo images
Three synthetic images are provided:
- `sample_a_1.png` and `sample_a_2.png` simulate the **same** writer.
- `sample_b_1.png` simulates a **different** writer.

---

## 🧰 Optional: Tiny CNN embedding (advanced)
We include a small CNN (`src/tiny_cnn.py`) to extract 128‑D embeddings. It is **off by default** (toggle in the sidebar). It uses random weights unless you train your own; it’s just a scaffold for students who want to experiment.

Train your own on public datasets like **IAM** or **CVL** and save weights to `cnn_weights.pth` (see docstring in `tiny_cnn.py`).

---

## 📏 Choosing a threshold
- Start at **0.5**.
- If you see too many “same” results, **increase** the threshold.
- If you see too many “different” results, **decrease** it slightly.

---

## 📝 Academic pointers
- Siamese networks with contrastive/triplet loss are standard for writer verification.
- Classical descriptors (HOG/LBP) still perform decently for *small* datasets.

---

## ⚖️ Ethics & disclaimer
This project is for **educational demonstration** only and not a forensic tool. Real-world writer identification requires expert methodology and validation.

---

## 📄 Citation (if used in your report)
> “An AI Handwriting Comparison system using HOG/LBP features and cosine similarity, wrapped in a Streamlit app.”
