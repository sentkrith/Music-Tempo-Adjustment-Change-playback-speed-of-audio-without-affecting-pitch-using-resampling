# Music-Tempo-Adjustment-Change-playback-speed-of-audio-without-affecting-pitch-using-resampling

# AIM
To develop a Python-based system that adjusts the tempo (playback speed) of an audio file without altering its pitch, using digital signal processing techniques such as time-stretching via resampling. The project enables users to upload any music or speech audio, modify its playback speed (faster or slower), and download the processed output while maintaining natural sound quality.

# APPARATUS REQUIRED 
Google Colab Notebook
# PROGRAM 
```python
# 🎵 Colab cell — Tempo adjustment (full-length playback, no 30s preview)
# Run this single cell in Google Colab

# 1) Install deps
!pip install -q librosa soundfile

# 2) Imports
import numpy as np
import librosa
import soundfile as sf
from google.colab import files
from IPython.display import Audio, display
import os

# 3) Upload and save file to disk
print("📤 Upload an audio file (wav, flac, mp3 ...)")
uploaded = files.upload()
if not uploaded:
    raise SystemExit("No file uploaded.")
in_filename = list(uploaded.keys())[0]
with open(in_filename, "wb") as f:
    f.write(uploaded[in_filename])
print(f"✅ Uploaded and saved to disk: {in_filename}")

# 4) Load audio (preserve channels)
data, sr = sf.read(in_filename, dtype='float32')   # data: (n,) or (n, channels)
if data.ndim == 1:
    data = data[:, np.newaxis]   # shape -> (n_samples, 1)

# 5) Set speed factor (edit as needed)
speed_factor = 1.25   # >1.0 faster, <1.0 slower

# 6) Time-stretch per-channel using librosa.effects.time_stretch
def time_stretch_per_channel(data, sr, speed_factor):
    stretched_channels = []
    for ch_idx in range(data.shape[1]):
        ch = data[:, ch_idx].astype(np.float32)
        stretched = librosa.effects.time_stretch(ch, rate=speed_factor)
        stretched_channels.append(stretched)
    max_len = max(len(c) for c in stretched_channels)
    stacked = np.stack([np.pad(c, (0, max_len - len(c)), mode='constant') for c in stretched_channels], axis=1)
    return stacked.astype(np.float32)

print("⏳ Processing full audio (this may take longer for large files)...")
stretched = time_stretch_per_channel(data, sr, speed_factor)  # (n_new, channels)

# 7) Save output to disk
out_filename = f"{os.path.splitext(in_filename)[0]}_tempo_{speed_factor:.2f}.wav"
sf.write(out_filename, stretched, sr, subtype='PCM_16')
print(f"💾 Saved output to: {out_filename}")

# 8) Play full original and full processed audio inline
# IPython.display.Audio accepts arrays shaped (channels, n_samples) for multi-channel audio
def play_full(orig, proc, sr):
    print("🎧 Original (full):")
    display(Audio(orig.T, rate=sr))
    print("🎧 Processed (full) — same pitch, tempo changed:")
    display(Audio(proc.T, rate=sr))

play_full(data, stretched, sr)

# 9) Trigger download (wrapped to avoid noisy errors)
try:
    files.download(out_filename)
except Exception as e:
    print("⚠️ Could not auto-download; file is saved in Colab's filesystem.")
    print("You can download from the Files pane or run `files.download(...)` manually.")
    print("Error:", e)

print("Done.")
```
# Original audio 
[Tamil_Thaai_Vazhthu.mp3](https://github.com/user-attachments/files/22945256/Tamil_Thaai_Vazhthu.mp3)

# Tempo Adjusted Audio (Pitch Preserved)
[Tamil_Thaai_Vazhthu_tempo_1.25.wav](https://github.com/user-attachments/files/22945264/Tamil_Thaai_Vazhthu_tempo_1.25.wav)


# Output
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/e9c854b1-9d93-42d6-bf91-998ec3f803e0" />

# Conclusion
The project successfully demonstrates how to modify the tempo of an audio file without altering its pitch using digital signal processing techniques in Python. By implementing the time-stretching method from the Librosa library, the system enables users to increase or decrease playback speed while maintaining natural tonal quality.

The waveform and audio outputs confirm that while the duration of the sound changes proportionally to the speed factor, the pitch and frequency characteristics remain constant, preserving the musical integrity of the track. This technique can be effectively applied in areas such as music production, speech analysis, remixing, and educational audio tools.
