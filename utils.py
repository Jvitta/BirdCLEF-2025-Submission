import cv2
import librosa
import numpy as np
import traceback
from pathlib import Path
import torch
from models.efficient_at.preprocess import AugmentMelSTFT

import warnings
warnings.filterwarnings("ignore")

# --- Worker Function for kaggle_inference.py ---
def _preprocess_audio_file_worker(audio_path_str, config):
    """Loads one audio file, generates all 5s specs and row IDs using AugmentMelSTFT."""
    audio_path = Path(audio_path_str)
    soundscape_id = audio_path.stem
    segment_specs = []
    segment_row_ids = []

    try:
        # Initialize AugmentMelSTFT here for each worker process call
        spectrogram_generator_worker = AugmentMelSTFT(
            n_mels=config.N_MELS,
            sr=config.FS,
            win_length=config.WIN_LENGTH,
            hopsize=config.HOP_LENGTH,
            n_fft=config.N_FFT,
            fmin=config.FMIN, # Ensure FMIN is in config, or pass None for default
            fmax=config.FMAX, # Let AugmentMelSTFT calculate base fmax from sr and fmax_aug_range
            freqm=0,          # Or config.FREQM if you make it configurable
            timem=0,          # Or config.TIMEM if you make it configurable
            fmin_aug_range=config.FMIN_AUG_RANGE, # Use config or default
            fmax_aug_range=config.FMAX_AUG_RANGE # Use config or default (e.g., 1000)
        )
        spectrogram_generator_worker.eval() # Set to eval mode for inference

        audio_data, _ = librosa.load(audio_path, sr=config.FS, mono=True)
        target_length_samples = int(config.TARGET_DURATION * config.FS)
        total_duration_samples = len(audio_data)
        num_segments = total_duration_samples // target_length_samples

        if num_segments == 0:
            return [], []

        for segment_idx in range(num_segments):
            start_sample = segment_idx * target_length_samples
            end_sample = start_sample + target_length_samples
            segment_audio = audio_data[start_sample:end_sample]

            if len(segment_audio) < target_length_samples:
                segment_audio = np.pad(segment_audio, (0, target_length_samples - len(segment_audio)), mode='constant')
            elif len(segment_audio) > target_length_samples:
                segment_audio = segment_audio[:target_length_samples]

            # Generate spectrogram using AugmentMelSTFT
            segment_audio_tensor = torch.from_numpy(segment_audio.astype(np.float32))
            with torch.no_grad():
                mel_spec_tensor = spectrogram_generator_worker(segment_audio_tensor.unsqueeze(0))
            mel_spec_numpy = mel_spec_tensor.squeeze(0).cpu().numpy()
            # mel_spec_numpy should now be (config.N_MELS, 500) or TARGET_SHAPE

            segment_specs.append(mel_spec_numpy.astype(np.float32))
            
            end_time_sec = (segment_idx + 1) * config.TARGET_DURATION
            row_id = f"{soundscape_id}_{int(end_time_sec)}"
            segment_row_ids.append(row_id)

        return segment_row_ids, segment_specs

    except Exception as e:
        print(f"\nError preprocessing worker {audio_path.name}: {e}\n{traceback.format_exc()}")
        return [], []

# def audio2melspec(audio_data, cfg): # This function is now obsolete
#     """Convert audio data to mel spectrogram"""
#     if np.isnan(audio_data).any():
#         mean_signal = np.nanmean(audio_data)
#         audio_data = np.nan_to_num(audio_data, nan=mean_signal)
# 
#     mel_spec = librosa.feature.melspectrogram(
#         y=audio_data,
#         sr=cfg.FS,
#         n_fft=cfg.N_FFT,
#         hop_length=cfg.HOP_LENGTH,
#         n_mels=cfg.N_MELS,
#         fmin=cfg.FMIN,
#         fmax=cfg.FMAX,
#         power=2.0
#     )
# 
#     mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#     mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
#     
#     return mel_spec_norm