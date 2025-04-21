# Music Genre Classifier 🎶

A TensorFlow 2 / Keras pipeline for classifying music genres from short audio clips.  
Includes:

- **Data preprocessing**: slicing `.wav` files into overlapping segments, computing and saving Mel‑spectrograms  
- **Baseline CNN** (150×150 spectrogram inputs)  
- **Hyperparameter‑tuned CNN** via KerasTuner  
- **Evaluation**: per-genre precision/recall, confusion matrices, training curves  
- **Streamlit app** for on‑the‑fly inference  

---

## 📁 Repository Layout

```
MUSIC_GENRE_CLASSIFIER/
├── .venv/                          # Python virtual environment
├── app/
│   └── app.py                     # Streamlit inference UI
├── data/
│   ├── genres_original/           # raw WAVs organized by genre
│   ├── spectrograms/              # optional images & arrays of spectrograms
│   └── splits/                    # NumPy arrays: X_train.npy, y_train.npy, etc.
├── models/
│   ├── baseline/                  # baseline CNN artifacts
│   │   ├── baseline.keras
│   │   └── baseline.weights.h5
│   └── hyperparameter_tuned/      # tuned-CNN artifacts
│       ├── hptune_best.weights.h5
│       └── hptune_full.keras
├── notebooks/
│   ├── 01_audio_exploration.ipynb
│   ├── 02_build_mel_spectrogram_dataset.ipynb
│   ├── 03_split_and_save_dataset.ipynb
│   ├── 04_baseline_model_training_and_evaluation.ipynb
│   └── 05_hyperparameter_model_training_and_tuning.ipynb
├── results/
│   ├── baseline_model/            # CSV reports & PNG plots for baseline
│   └── hyperparameter_model/      # CSV reports & PNG plots for tuned model
├── utils/
│   ├── audio_utils.py             # segment_audio() & helper functions
│   └── __pycache__/
├── requirements.txt               # all Python package dependencies       
└── README.md                      # this file
```

---

## 🚀 Getting Started

### 1. Clone & create environment

```bash
git clone https://github.com/yourname/MUSIC_GENRE_CLASSIFIER.git
cd MUSIC_GENRE_CLASSIFIER
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare directories

Optionally run:

```bash
python setup_dirs.py
```

This will create the following empty folders if they don’t exist:
`data/genres_original/`, `data/spectrograms/`, `data/splits/`,  
`models/baseline/`, `models/hyperparameter_tuned/`, `results/…`

### 3. Add your raw audio

Place your `.wav` files under:

```
data/genres_original/
    blues/
    classical/
    country/
    … (one folder per genre)
```

### 4. Preprocess & split

Run **Notebook 02_build_mel_spectrogram_dataset.ipynb** to

- Segment audio into overlapping clips  
- Compute and save mel-spectrogram arrays/images  
- Generate `data/splits/X_train.npy`, `y_train.npy`, etc.

### 5. Train Baseline Model

Run **Notebook 04_baseline_model_training_and_evaluation.ipynb** to

- Load `data/splits/*.npy`  
- Build & train the baseline CNN  
- Save best weights & full model to `models/baseline/`  
- Export metrics & plots to `results/baseline_model/`

### 6. Hyperparameter Tuning

Run **Notebook 05_hyperparameter_model_training_and_tuning.ipynb** to

- Define a tunable CNN architecture  
- Quick search on a 2 k subset  
- Retrain best configuration on full data  
- Save tuned model to `models/hyperparameter_tuned/`  
- Export metrics & plots to `results/hyperparameter_model/`

### 7. Streamlit App

```bash
streamlit run app/app.py
```

Upload an audio file and view:
- Playback  
- Predicted genre with % confidence  
- Segment-level vote distribution chart  

---

## 📊 Results

Explore output files in `results/baseline_model/` and `results/hyperparameter_model/`:
- `_test_metrics.txt`  
- `_classification_report.csv`  
- `_confusion_matrix.png`  
- `_precision_per_genre.png`  
- `_recall_per_genre.png`  
- `_training_curves.png`  

---
## Author
Jivesh Dhakate
MSc Computer Science, University College Dublin

