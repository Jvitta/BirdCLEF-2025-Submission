import os
import cv2
import math
import time
import librosa
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import IPython.display as ipd

import torch
import warnings
import io
import zipfile
warnings.filterwarnings("ignore")

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

# --- Debug Helper for Chunk Visualization/Playback ---
def _plot_debug_chunk_waveform(audio_chunk, sr, title):
    """Helper function to plot waveform and provide audio playback for a single audio chunk."""
    if audio_chunk is None or len(audio_chunk) == 0:
        print(f"Debug Chunk Plot: Skipping plot/audio for '{title}' - chunk is empty.")
        return
    try:
        print(f"--- Debugging Chunk: {title} ---")
        # Plotting
        plt.figure(figsize=(12, 2.5)) # Smaller figure for chunks
        librosa.display.waveshow(audio_chunk, sr=sr)
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
        # Audio Playback
        display(ipd.Audio(data=audio_chunk, rate=sr))
        print("------------------------------------")
        
    except Exception as e:
        print(f"Debug Chunk Plot/Audio: Error processing waveform for '{title}': {e}")
# --- End Debug Helper ---

def process_audio_file(filepath, filename, cfg, fabio_intervals, vad_intervals, max_chunks_allowed=None):
    """Process a single audio file, remove speech/isolate sound using intervals,
       then chunk the relevant audio into 5-second segments and generate
       a mel spectrogram for each chunk.
       Special handling for the last chunk based on whether it's the only chunk.
       Optionally limits the number of chunks returned per file.

       Args:
           filepath (str): Path to the audio file.
           filename (str): Name of the audio file.
           cfg (Config): Configuration object.
           fabio_intervals (dict): Dictionary of Fabio intervals.
           vad_intervals (dict): Dictionary of VAD intervals.
           max_chunks_allowed (int, optional): Maximum number of spectrogram chunks to return for this file. Defaults to None (no limit).

       Returns a list of spectrograms, or an empty list if no valid chunks found.
    """
    spectrogram_chunks = []
    try:
        audio_data, sr = librosa.load(filepath, sr=cfg.FS)
        if sr != cfg.FS:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=cfg.FS)

        relevant_audio = None
        target_samples = int(cfg.TARGET_DURATION * cfg.FS)
        min_total_audio_samples = int(0.5 * cfg.FS) # 0.5 seconds
        min_last_chunk_multi_samples = int(3.0 * cfg.FS) # 3 seconds

        if filename in fabio_intervals:
            start_time, stop_time = fabio_intervals[filename]
            start_idx = int(start_time * cfg.FS)
            end_idx = int(stop_time * cfg.FS)
            start_idx = max(0, start_idx)
            end_idx = min(len(audio_data), end_idx)
            if start_idx < end_idx:
                relevant_audio = audio_data[start_idx:end_idx]
            else:
                print(f"Warning: Invalid Fabio interval for {filename}. Using full audio.")
                relevant_audio = audio_data
        elif filepath in vad_intervals:
            speech_timestamps = vad_intervals[filepath]

            non_speech_segments = []
            current_pos_sec = 0.0
            audio_duration_sec = len(audio_data) / cfg.FS

            speech_timestamps.sort(key=lambda x: x['start'])

            for segment in speech_timestamps:
                start_speech_sec = segment['start']
                end_speech_sec = segment['end']

                # Add non-speech segment before the current speech segment
                if start_speech_sec > current_pos_sec:
                    start_idx = int(current_pos_sec * cfg.FS)
                    end_idx = int(start_speech_sec * cfg.FS)
                    # Ensure indices are valid before slicing
                    start_idx = max(0, start_idx)
                    end_idx = min(len(audio_data), end_idx)
                    if start_idx < end_idx:
                         non_speech_segments.append(audio_data[start_idx:end_idx])

                current_pos_sec = max(current_pos_sec, end_speech_sec)

            # Add any remaining non-speech segment after the last speech segment
            if current_pos_sec < audio_duration_sec:
                start_idx = int(current_pos_sec * cfg.FS)
                # Ensure indices are valid before slicing
                start_idx = max(0, start_idx)
                if start_idx < len(audio_data):
                    non_speech_segments.append(audio_data[start_idx:])

            if non_speech_segments:
                # Ensure segments are numpy arrays before concatenating
                non_speech_segments = [np.asarray(seg) for seg in non_speech_segments if len(seg) > 0]
                if non_speech_segments: # Check if list is not empty after filtering
                     relevant_audio = np.concatenate(non_speech_segments)
                else:
                     print(f"Warning: VAD processing resulted in only empty segments for {filename}. Using full audio.")
                     relevant_audio = audio_data # Fallback
            else:
                 # If VAD removed everything (e.g., all speech or silence)
                 print(f"Warning: VAD removed all segments for {filename}. Using full audio as fallback.")
                 relevant_audio = audio_data # Fallback

        # 3. Fallback: No specific intervals found
        else:
            relevant_audio = audio_data

        # Check if relevant_audio meets the absolute minimum length (0.5s)
        if relevant_audio is None or len(relevant_audio) < min_total_audio_samples:
            print(f"Warning: Relevant audio for {filename} too short ({len(relevant_audio)/sr if relevant_audio is not None else 0:.2f}s < {min_total_audio_samples/sr:.2f}s) or None. Skipping file.")
            return [] # Return empty list

        # 4. Chunk the relevant_audio into segments
        num_chunks = math.ceil(len(relevant_audio) / target_samples)

        # Determine the number of chunks to actually process based on the limit
        chunks_to_process = num_chunks
        if max_chunks_allowed is not None:
            chunks_to_process = min(num_chunks, max_chunks_allowed)

        # Loop only through the required number of chunks
        for i in range(chunks_to_process): # Modified loop range
            start = i * target_samples
            end = start + target_samples
            chunk = relevant_audio[start:end]
            chunk_len = len(chunk)
            # Determine if this chunk would have been the last one *if all chunks were processed*
            # This is needed for the padding logic
            is_potentially_last_chunk = (i == num_chunks - 1)

            processed_chunk = None # Variable to hold the chunk after padding/checking

            # Handle potentially short chunks
            if chunk_len < target_samples:
                 # Check if it's the ONLY chunk produced AND meets the 0.5s threshold
                if is_potentially_last_chunk and num_chunks == 1 and chunk_len >= min_total_audio_samples:
                     padding = target_samples - chunk_len
                     processed_chunk = np.pad(chunk, (0, padding), mode='constant')
                # Check if it's the LAST chunk of MULTIPLE chunks AND meets the 3s threshold
                elif is_potentially_last_chunk and num_chunks > 1 and chunk_len >= min_last_chunk_multi_samples:
                     padding = target_samples - chunk_len
                     processed_chunk = np.pad(chunk, (0, padding), mode='constant')
                # Otherwise (it's short and doesn't meet criteria), skip it
                else:
                     # Skip chunks that are too short unless they are the last and meet padding criteria
                     # This check needs to be careful - we only skip if it's short AND not the last valid one
                     if not is_potentially_last_chunk: # Only skip if it's not the potentially last chunk
                        # print(f"    Skipping intermediate short chunk {i} for {filename} (length {chunk_len/sr:.2f}s)")
                        continue
                     else:
                         # If it IS the potentially last chunk but didn't meet padding criteria, skip
                         # print(f"    Skipping potentially last chunk {i} for {filename} (length {chunk_len/sr:.2f}s did not meet padding criteria)")
                         continue
            else:
                # Chunk was already full length
                processed_chunk = chunk

            # Should always have a valid chunk here unless skipped above
            if processed_chunk is None or len(processed_chunk) != target_samples:
                print(f"Warning: Internal logic error - chunk {i} for {filename} processing resulted in invalid length. Skipping.")
                continue

            # --- Debug Plot/Audio for the specific chunk ---
            if cfg.debug_preprocessing_mode:
                 # Determine padding status based on the logic above
                 padded = (
                     (chunk_len < target_samples) and
                     (
                         (is_potentially_last_chunk and num_chunks == 1 and chunk_len >= min_total_audio_samples) or
                         (is_potentially_last_chunk and num_chunks > 1 and chunk_len >= min_last_chunk_multi_samples)
                     )
                 )
                 chunk_title = f"Chunk {i} for {filename} (Orig Len: {chunk_len/sr:.2f}s, Padded: {padded})"
                 _plot_debug_chunk_waveform(processed_chunk, sr, chunk_title)
            # --- End Debug Plot/Audio ---

            # 5. Generate Mel Spectrogram for the chunk
            mel_spec = audio2melspec(processed_chunk, cfg)

            # 6. Resize Spectrogram
            if mel_spec.shape[1] != cfg.TARGET_SHAPE[1] or mel_spec.shape[0] != cfg.TARGET_SHAPE[0]:
                 mel_spec = cv2.resize(mel_spec, (cfg.TARGET_SHAPE[1], cfg.TARGET_SHAPE[0]), interpolation=cv2.INTER_LINEAR)

            spectrogram_chunks.append(mel_spec.astype(np.float16))

        return spectrogram_chunks

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return [] # Return empty list on error

def generate_spectrograms(df, cfg, fabio_intervals, vad_intervals, zip_filepath):
    """Generate spectrograms, write each chunk directly to a zip archive, and return list of keys.
       Limits the number of chunks per file based on class distribution.

    Args:
        df (pd.DataFrame): DataFrame with audio metadata (must include 'filepath', 'filename', 'primary_label').
        cfg (Config): Configuration object.
        fabio_intervals (dict): Dictionary of Fabio intervals.
        vad_intervals (dict): Dictionary of VAD intervals.
        zip_filepath (str): Path to the final zip archive to create.

    Returns:
        list: A list of chunk_keys corresponding to successfully written spectrogram chunks.
    """
    print(f"Generating mel spectrogram chunks and writing directly to: {zip_filepath}")
    start_time = time.time()
    os.makedirs(os.path.dirname(zip_filepath), exist_ok=True)

    # --- Calculate Max Chunks per File based on Class Distribution ---
    max_chunks_per_class = 1000
    metadata_path = '/kaggle/input/train-metadata-chunked/train_metadata_chunked_full.csv'
    print(f"Reading full metadata from {metadata_path} to calculate chunk limits...")
    try:
        full_meta_df = pd.read_csv(metadata_path)
        files_per_class = full_meta_df.groupby('primary_label')['filename'].nunique()
        chunks_per_class = full_meta_df['primary_label'].value_counts()

        max_chunks_per_file_dict = {}
        default_max = max_chunks_per_class
        print(f"Calculating max chunks per file (target max per class: {max_chunks_per_class})...")
        for label, num_files in files_per_class.items():
            num_chunks = chunks_per_class.get(label, 0)


            if num_chunks < max_chunks_per_class:
                 max_chunks_per_file_dict[label] = max_chunks_per_class
            elif num_files > 0: 
                 max_chunks_for_file = max(1, int(max_chunks_per_class / num_files))
                 max_chunks_per_file_dict[label] = max_chunks_for_file
            else:
                 max_chunks_per_file_dict[label] = default_max

        if max_chunks_per_file_dict:
            numeric_limits = [v for v in max_chunks_per_file_dict.values() if isinstance(v, (int, float))]
            if numeric_limits:
                overall_default_max = math.ceil(np.mean(numeric_limits))
            else:
                overall_default_max = default_max
        else:
            overall_default_max = default_max
        print(f"Calculated per-file chunk limits. Default limit for missing labels: {overall_default_max}")

    except Exception as e:
        print(f"Error reading or processing {metadata_path}: {e}. Cannot apply class-based chunk limiting.")
        max_chunks_per_file_dict = {}
        overall_default_max = None

    saved_chunk_keys = []
    errors = []
    total_chunks_generated = 0
    chunks_limited_count = 0

    try:
        with zipfile.ZipFile(zip_filepath, 'w', compression=zipfile.ZIP_DEFLATED) as zip_ref:
            for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing Audio & Writing to Zip"):
                try:
                    filepath = row["filepath"]
                    filename = row["filename"]
                    primary_label = row["primary_label"] # Get label for the current file

                    # Determine max chunks allowed for this file's label
                    max_chunks_for_this_file = None # Default to None (no limit) if dict is empty or label not found
                    if max_chunks_per_file_dict: # Only apply if calculation was successful
                         max_chunks_for_this_file = max_chunks_per_file_dict.get(primary_label, overall_default_max)

                    # Call process_audio_file with the calculated limit
                    spectrogram_chunks = process_audio_file(
                        filepath, filename, cfg, fabio_intervals, vad_intervals,
                        max_chunks_allowed=max_chunks_for_this_file # Pass the limit
                    )

                    if spectrogram_chunks:
                        # Check if limiting occurred (useful for logging/debugging)
                        # Note: process_audio_file internally calculates num_chunks based on *relevant_audio* length.
                        # We need to know the *potential* number of chunks *before* truncation to log accurately if limiting happened.
                        # This requires a slight modification or approximation. Let's assume if max_chunks_for_this_file was applied
                        # and the returned number equals it, limiting *might* have happened. A more precise check
                        # would require process_audio_file to return the original potential count.
                        # For now, let's increment if a limit was passed and chunks were returned.
                        if max_chunks_for_this_file is not None and len(spectrogram_chunks) == max_chunks_for_this_file:
                            chunks_limited_count += 1

                        for chunk_idx, chunk_spec in enumerate(spectrogram_chunks):
                            chunk_key = f"{filename}_chunk{chunk_idx}"
                            arcname = f"{chunk_key}.npy"

                            try:
                                with io.BytesIO() as buffer:
                                    np.save(buffer, chunk_spec, allow_pickle=False)
                                    bytes_data = buffer.getvalue()

                                zip_ref.writestr(arcname, bytes_data)
                                saved_chunk_keys.append(chunk_key)
                                total_chunks_generated += 1

                            except Exception as write_e:
                                print(f"Error writing chunk {chunk_key} ({arcname}) to zip: {write_e}")
                                errors.append((filepath, f"Writing chunk {chunk_key}: {write_e}"))

                    else:
                        # Only add error if processing didn't just return an empty list due to skipping short audio
                        # The process_audio_file function already prints warnings for skipped files.
                        # Let's refine this: only add error if spectrogram_chunks is None explicitly?
                        # Current process_audio_file returns [] on error or skip.
                        # We'll keep the logic as is for now, assuming [] means either skip or error handled internally.
                        errors.append((filepath, "Processing returned no valid chunks (or file skipped)"))

                except Exception as e:
                    print(f"Error in generate_spectrograms loop for {filepath}: {e}")
                    errors.append((filepath, str(e)))

    except Exception as zip_e:
        print(f"Fatal error creating or writing to zip file {zip_filepath}: {zip_e}")
        return []

    end_time = time.time()
    print(f"\nDirect zip writing completed in {end_time - start_time:.2f} seconds")
    print(f"Successfully generated and wrote {total_chunks_generated} spectrogram chunks to {zip_filepath}.")
    if max_chunks_per_file_dict: # Only print if limiting was attempted
        print(f"Chunk limiting applied based on class distribution. Files potentially limited: {chunks_limited_count}")
    print(f"Failed or skipped {len(errors)} files/chunks during processing/writing")
    if errors:
        print("\nFiles/Chunks with processing or writing errors/skips:")
        for err_file, reason in errors[:10]: # Show first 10 errors/skips
            print(f"- {err_file}: {reason}")

    return saved_chunk_keys