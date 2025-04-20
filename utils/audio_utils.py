import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def segment_audio(y, sr, segment_duration, overlap):
    hop_duration = segment_duration - overlap
    segment_samples = int(segment_duration * sr)
    hop_samples = int(hop_duration * sr)

    segments = []
    num_segments = int((len(y) - segment_samples) / hop_samples) + 1

    for i in range(num_segments):
        start = i * hop_samples
        end = start + segment_samples

        if end > len(y):
            break

        segment = y[start:end]
        segments.append(segment)

    return segments

def plot_waveshow_of_segments(segments, sr, cols=3):
    num = len(segments)
    rows = (num + cols - 1) // cols

    plt.figure(figsize=(cols * 5, rows * 3))

    for i, seg in enumerate(segments):
        duration = len(seg) / sr
        t = np.linspace(0, duration, len(seg))

        plt.subplot(rows, cols, i + 1)
        librosa.display.waveshow(seg, sr=sr)
        plt.title(f"Segment {i+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)

    plt.suptitle("Waveform of All Segments", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def generate_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128, display=True):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)

    if display:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length,
                                 x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Mel Spectrogram")
        plt.tight_layout()
        plt.show()

    return S_dB

def plot_mel_spectrograms_of_segments(segments, sr, cols=3):
    num = len(segments)
    rows = (num + cols - 1) // cols

    plt.figure(figsize=(cols * 5, rows * 4))

    for i, seg in enumerate(segments):
        S = librosa.feature.melspectrogram(y=seg, sr=sr, n_fft=2048,
                                           hop_length=512, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.subplot(rows, cols, i + 1)
        librosa.display.specshow(S_dB, sr=sr, hop_length=512,
                                 x_axis='time', y_axis='mel')
        plt.title(f'Segment {i+1}')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()

    plt.suptitle("Mel Spectrograms of All Segments", fontsize=16, y=1.02)
    plt.show()

def plot_segment_overlay(y, sr, segment_duration, hop):
    t = np.linspace(0, len(y) / sr, len(y))
    plt.figure(figsize=(14, 4))
    plt.plot(t, y, label="Waveform", alpha=0.5)
    num_segments = int((len(y) - int(segment_duration * sr)) / int(hop * sr)) + 1
    for i in range(num_segments):
        start = i * hop
        end = start + segment_duration
        plt.axvspan(start, end, color='red', alpha=0.2)
    plt.title(f"Segment Overlay ({segment_duration}s / Hop {hop}s)")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.show()
