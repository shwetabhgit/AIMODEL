import os
import torch
import torchaudio
from scipy.spatial.distance import cosine
import urllib.request

def download_sample_audios(stored_audio_dir):
    os.makedirs(stored_audio_dir, exist_ok=True)
    samples = {
        "sample1.wav": "https://github.com/mdeff/fma/raw/master/data/fma_small/000/000002.mp3",
        "sample2.wav": "https://github.com/mdeff/fma/raw/master/data/fma_small/000/000003.mp3",
        "sample3.wav": "https://github.com/mdeff/fma/raw/master/data/fma_small/000/000005.mp3",
    }
    for fname, url in samples.items():
        wav_path = os.path.join(stored_audio_dir, fname)
        if not os.path.exists(wav_path):
            mp3_path = wav_path.replace('.wav', '.mp3')
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(url, mp3_path)
            # Convert mp3 to wav using torchaudio
            waveform, sr = torchaudio.load(mp3_path)
            torchaudio.save(wav_path, waveform, sr)
            os.remove(mp3_path)

def extract_features(filepath):
    waveform, sample_rate = torchaudio.load(filepath)
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # Extract MFCC features
    mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=40)(waveform)
    # Take mean over time axis
    return mfcc.mean(dim=2).squeeze().numpy()

def find_similar_audio(uploaded_path, stored_dir, top_k=3):
    uploaded_feat = extract_features(uploaded_path)
    similarities = []
    for fname in os.listdir(stored_dir):
        if fname.endswith('.wav'):
            stored_path = os.path.join(stored_dir, fname)
            stored_feat = extract_features(stored_path)
            sim = 1 - cosine(uploaded_feat, stored_feat)
            similarities.append((fname, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

if __name__ == "__main__":
    # Directory containing stored audio files (all .wav)
    stored_audio_dir = "stored_audios"
    # Download sample audio files if not present
    download_sample_audios(stored_audio_dir)

    # Path to the uploaded audio file (for demo, use one of the samples)
    uploaded_audio_path = os.path.join(stored_audio_dir, "sample1.wav")

    # Find top 3 most similar audio files
    results = find_similar_audio(uploaded_audio_path, stored_audio_dir, top_k=3)
    print("Most similar audio files:")
    for fname, score in results:
        print(f"{fname}: similarity={score:.4f}")