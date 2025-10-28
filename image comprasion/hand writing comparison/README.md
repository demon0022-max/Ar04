 AI Handwriting Comparison (Minor Project)

An end‑to‑end Streamlit app that compares two handwriting images and predicts if they are from the same writer or different writers.

 Features
- Upload two images (JPG/PNG).
- Robust preprocessing: grayscale, binarize, deskew.
- Feature extraction: **HOG**, **LBP**, and simple stroke/texture stats.
- Similarity with **cosine distance** + adjustable threshold.
- Visual feedback: similarity score, decision, and intermediate previews.
- Optional: toggle a lightweight CNN embedding (if PyTorch installed) as an add‑on.

  PROJECT STRUCTURE
handwriting_comparator/
│
├── app.py                 
├── requirements.txt
├── README.md
└── src/
├── features.py        
├── compare.py         
└── cnn.py        

 Using the app
1. Upload **two** handwriting images (cropped word/line/page all work).
2. Optionally adjust threshold (default tuned ≈ 0.45–0.55 for our feature mix).
3. Click Compare to see:
   - Similarity score (0 to 1 where 1 is identical).
   - Decision: **Same writer** or **Different writer**.
   - Preprocessing previews of each image.

How it works
- Preprocessing
  - grayscale → median denoise → adaptive binarize
  - **deskew** via Hough-based angle estimation
  - resize to 256×256, center pad
- Features
  - HOG (oriented gradients texture)
  - LBP histogram (micro‑texture)
  - Simple stats (ink ratio, stroke density, edge density, contour complexity)
- Compare
  - Normalize vectors → **cosine similarity**
  - Decision by threshold (user‑adjustable)


