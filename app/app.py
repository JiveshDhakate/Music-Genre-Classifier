import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tempfile
from collections import Counter

# ─── 0) Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music Genre Classifier 🎶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── 1) Load Trained Model (.keras format) ────────────────────────────────
MODEL_PATH = (
    "/Users/jiveshdhakate/Documents/UCD Sem 2/Deep Learning/"
    "Project/music_genre_classifier/models/hyperparameter_tuned/"
    "hptune_full.keras"
)
model = tf.keras.models.load_model(MODEL_PATH)

# ─── 2) Constants (must match training) ───────────────────────────────────
CLASS_LABELS = [
    'blues','classical','country','disco','hiphop',
    'jazz','metal','pop','reggae','rock'
]
SR           = 22050
SEG_DUR      = 3.0
HOP_DUR      = 1.5
N_FFT        = 2048
HOP_LENGTH   = 512
N_MELS       = 128
TARGET_HW    = (150, 150)   # height × width for CNN

# ─── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
st.sidebar.markdown(f"- **Sample rate:** {SR}")
st.sidebar.markdown(f"- **Segment dur:** {SEG_DUR}s")
st.sidebar.markdown(f"- **Hop dur:** {HOP_DUR}s")

# ─── Main UI ──────────────────────────────────────────────────────────────
st.title("🎶 Music Genre Classifier")
st.write(
    "Upload an audio file (WAV or MP3). "
    "We'll split into 3 s segments, run our CNN, "
    "and then vote on the overall genre."
)

uploaded = st.file_uploader("Choose a WAV or MP3 file", type=["wav","mp3"])
if not uploaded:
    st.info("Please upload a file to get started.")
    st.stop()

# ─── 3) Load & play the audio ─────────────────────────────────────────────
with tempfile.NamedTemporaryFile(suffix=uploaded.name[-4:], delete=False) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

y, _ = librosa.load(tmp_path, sr=SR)
duration = len(y) / SR

st.subheader("🔊 Audio Preview")
st.audio(tmp_path, format=f"audio/{uploaded.name.split('.')[-1]}")
st.caption(f"Duration: {duration:.1f} s")

# ─── 4) Segment → preprocess → predict ───────────────────────────────────
votes = []
for start in np.arange(0, duration - SEG_DUR + 1e-6, HOP_DUR):
    a, b = int(start*SR), int((start+SEG_DUR)*SR)
    seg = y[a:b]
    if np.max(np.abs(seg)) < 0.01:
        continue

    # mel‑spectrogram → dB
    S = librosa.feature.melspectrogram(
        y=seg, sr=SR, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # resize to TARGET_HW & z‑score
    img = tf.image.resize(
        S_db[..., np.newaxis], TARGET_HW, method="bilinear"
    ).numpy().squeeze()
    img = (img - img.mean()) / (img.std() + 1e-8)

    # predict top‐3, collect votes
    inp = img[np.newaxis, ..., np.newaxis].astype(np.float32)
    probs = model.predict(inp, verbose=0)[0]
    top3  = np.argsort(probs)[-3:][::-1]
    votes.extend([CLASS_LABELS[i] for i in top3])

# ─── 5) Aggregate votes ──────────────────────────────────────────────────
counter = Counter(votes)
total   = sum(counter.values())
# Top‐1 genre and its percentage
top_genre, top_count = counter.most_common(1)[0]
top_pct = top_count / total * 100

# Build distribution for bar chart
dist = {g: cnt/total*100 for g, cnt in counter.items()}

# ─── 6) Display results ─────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.metric("🏆 Predicted Genre", top_genre.capitalize(), f"{top_pct:.1f}%")
    st.markdown("**Vote distribution by segment:**")
    # Create a DataFrame for the vote table
    df_votes = pd.DataFrame({
        "Genre":      [g.capitalize() for g in counter.keys()],
        "Percentage": [cnt/total*100 for cnt in counter.values()]
    })
    df_votes["Percentage"] = df_votes["Percentage"].map(lambda x: f"{x:.1f}%")
    st.dataframe(df_votes, use_container_width=True)

with col2:
    st.markdown("### 📊 Genre Vote Distribution")
    st.bar_chart(dist)

st.markdown(
    "---\n"
    "Model trained with hyperparameter tuning for maximal validation accuracy."
)
