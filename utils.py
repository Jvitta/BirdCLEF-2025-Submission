import cv2
import librosa
import numpy as np
import traceback
from pathlib import Path
import torch
from models.efficient_at.preprocess import AugmentMelSTFT
import math

import warnings
warnings.filterwarnings("ignore")

# --- Worker Function for kaggle_inference.py ---
def _preprocess_audio_file_worker(audio_path_str, config):
    """
    Loads one audio file. For each of the 12 expected 5-second prediction intervals
    in a 60-second soundscape, it extracts a 10-second audio segment
    (config.TARGET_DURATION) centered on that 5s interval. If the 10s segment goes
    out of bounds (e.g., due to the soundscape being slightly shorter than 60s),
    it's padded on the right with zeros. Spectrograms are then generated from these
    10s audio segments.
    """
    audio_path = Path(audio_path_str)
    soundscape_id = audio_path.stem
    segment_specs = []
    segment_row_ids = []

    try:
        spectrogram_generator_worker = AugmentMelSTFT(
            n_mels=config.N_MELS,
            sr=config.FS,
            win_length=config.WIN_LENGTH,
            hopsize=config.HOP_LENGTH,
            n_fft=config.N_FFT,
            fmin=config.FMIN,
            fmax=config.FMAX,
            freqm=0, timem=0,
            fmin_aug_range=config.FMIN_AUG_RANGE,
            fmax_aug_range=config.FMAX_AUG_RANGE
        )
        spectrogram_generator_worker.eval()

        audio_data, _ = librosa.load(str(audio_path), sr=config.FS, mono=True) 
        
        total_audio_samples = len(audio_data)
        _total_audio_duration_sec = total_audio_samples / config.FS # For clarity

        prediction_interval_sec = 5.0 # We generate a prediction for every 5s window
        model_input_duration_sec = config.TARGET_DURATION 
        model_input_samples = int(model_input_duration_sec * config.FS)

        num_prediction_intervals = 12 # Hardcode to 12 for 60s soundscapes

        for i in range(num_prediction_intervals):
            current_5s_target_start_sec = i * prediction_interval_sec
            current_5s_target_end_sec_for_rowid = (i + 1) * prediction_interval_sec 

            segment_audio_10s = None

            if i == 0: # First chunk: Take first 7s of audio
                actual_extraction_start_samples = 0
                actual_extraction_end_samples = min(model_input_samples, total_audio_samples)
                segment_audio_10s = audio_data[actual_extraction_start_samples:actual_extraction_end_samples]
            
            elif i == num_prediction_intervals - 1: # Last chunk: Take last 7s of audio
                actual_extraction_start_samples = max(0, total_audio_samples - model_input_samples)
                actual_extraction_end_samples = total_audio_samples
                segment_audio_10s = audio_data[actual_extraction_start_samples:actual_extraction_end_samples]

            else: # Middle chunks: Use centered window, clipped and padded
                center_of_5s_target_sec = current_5s_target_start_sec + (prediction_interval_sec / 2.0)
                ideal_7s_extraction_start_sec = center_of_5s_target_sec - (model_input_duration_sec / 2.0)
                
                actual_extraction_start_sec = max(0, ideal_7s_extraction_start_sec)
                actual_extraction_start_samples = int(round(actual_extraction_start_sec * config.FS))
                
                ideal_7s_extraction_end_sec = ideal_7s_extraction_start_sec + model_input_duration_sec
                actual_slice_end_sec = min(ideal_7s_extraction_end_sec, _total_audio_duration_sec)
                actual_extraction_end_samples = int(round(actual_slice_end_sec * config.FS))
                actual_extraction_end_samples = max(actual_extraction_start_samples, actual_extraction_end_samples)
                
                segment_audio_10s = audio_data[actual_extraction_start_samples:actual_extraction_end_samples]

            # Pad ON THE RIGHT if the extracted segment is shorter than config.TARGET_DURATION (e.g. 7s)
            current_len_samples = len(segment_audio_10s)
            if current_len_samples < model_input_samples: # model_input_samples is config.TARGET_DURATION * FS
                padding_needed = model_input_samples - current_len_samples
                segment_audio_10s = np.pad(segment_audio_10s, (0, padding_needed), mode='constant', constant_values=0)
            elif current_len_samples > model_input_samples: 
                segment_audio_10s = segment_audio_10s[:model_input_samples]
            # At this point, segment_audio_10s is exactly model_input_samples (e.g. 7s from config.TARGET_DURATION) long.
            # Let's call this our base audio for the next step.
            base_audio_for_final_processing = segment_audio_10s

            # --- New logic: Append middle 3s of the base_audio_for_final_processing to itself ---
            model_target_total_audio_samples = int(10.0 * config.FS)
            
            samples_in_base_audio = len(base_audio_for_final_processing) # Should be 7s * FS

            # Define the duration of the middle chunk to extract and append
            middle_chunk_to_append_duration_sec = 3.0
            middle_chunk_to_append_samples = int(middle_chunk_to_append_duration_sec * config.FS)

            # Extract the middle_chunk_to_append_samples from base_audio_for_final_processing
            start_of_middle_in_base = max(0, (samples_in_base_audio - middle_chunk_to_append_samples) // 2)
            end_of_middle_in_base = start_of_middle_in_base + middle_chunk_to_append_samples
            
            middle_chunk = base_audio_for_final_processing[start_of_middle_in_base : min(end_of_middle_in_base, samples_in_base_audio)]
            
            # Concatenate the base audio (7s) with its (raw extracted) middle chunk (approx 3s)
            concatenated_audio = np.concatenate((base_audio_for_final_processing, middle_chunk))
            
            # Ensure the final concatenated_audio matches the model_target_total_audio_samples precisely
            current_concat_len = len(concatenated_audio)
            if current_concat_len < model_target_total_audio_samples:
                concatenated_audio = np.pad(concatenated_audio, (0, model_target_total_audio_samples - current_concat_len), mode='constant', constant_values=0)
            elif current_concat_len > model_target_total_audio_samples:
                concatenated_audio = concatenated_audio[:model_target_total_audio_samples]
            
            segment_audio_10s = concatenated_audio # This is now the audio to be converted to spectrogram
            # --- End of new logic ---

            # Generate spectrogram from the audio chunk
            segment_audio_tensor = torch.from_numpy(segment_audio_10s.astype(np.float32))
            with torch.no_grad():
                mel_spec_tensor = spectrogram_generator_worker(segment_audio_tensor.unsqueeze(0))
            mel_spec_numpy = mel_spec_tensor.squeeze(0).cpu().numpy()

            segment_specs.append(mel_spec_numpy.astype(np.float32))
            
            row_id = f"{soundscape_id}_{int(current_5s_target_end_sec_for_rowid)}"
            segment_row_ids.append(row_id)

        return segment_row_ids, segment_specs

    except Exception as e:
        print(f"\nError preprocessing worker {audio_path.name}: {e}\n{traceback.format_exc()}")
        return [], []