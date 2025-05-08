import cv2
import librosa
import numpy as np
import traceback
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

# --- Worker Function for kaggle_inference.py ---
def _preprocess_audio_file_worker(audio_path_str, config):
    """Loads one audio file, generates all 5s specs and row IDs."""
    audio_path = Path(audio_path_str)
    soundscape_id = audio_path.stem
    segment_specs = []
    segment_row_ids = []

    try:
        audio_data, _ = librosa.load(audio_path, sr=config.FS, mono=True)
        target_length_samples = int(config.TARGET_DURATION * config.FS)
        total_duration_samples = len(audio_data)
        num_segments = total_duration_samples // target_length_samples

        # Skip short files
        if num_segments == 0:
            return [], []

        for segment_idx in range(num_segments):
            start_sample = segment_idx * target_length_samples
            end_sample = start_sample + target_length_samples
            segment_audio = audio_data[start_sample:end_sample]

            # Ensure the segment is exactly target_length_samples long
            if len(segment_audio) < target_length_samples:
                segment_audio = np.pad(segment_audio, (0, target_length_samples - len(segment_audio)), mode='constant')
            elif len(segment_audio) > target_length_samples:
                segment_audio = segment_audio[:target_length_samples]

            # Generate normalized mel spectrogram using the utility function
            mel_spec_norm = audio2melspec(segment_audio, config)

            # Resize the normalized spectrogram
            if mel_spec_norm.shape != config.TARGET_SHAPE:
                mel_spec_resized = cv2.resize(mel_spec_norm, (config.TARGET_SHAPE[1], config.TARGET_SHAPE[0]), interpolation=cv2.INTER_LINEAR)
            else:
                mel_spec_resized = mel_spec_norm

            segment_specs.append(mel_spec_resized.astype(np.float32))
            
            end_time_sec = (segment_idx + 1) * config.TARGET_DURATION
            row_id = f"{soundscape_id}_{int(end_time_sec)}"
            segment_row_ids.append(row_id)

        return segment_row_ids, segment_specs

    except Exception as e:
        print(f"\nError preprocessing worker {audio_path.name}: {e}\n{traceback.format_exc()}")
        return [], []

def audio2melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm