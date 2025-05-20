import pickle
import os
import sys
import pandas as pd
from collections import Counter
import numpy as np 
from tqdm import tqdm
import librosa # For getting audio duration
import random
import shutil # For copying files

# Add project root to sys.path to allow importing config
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from config import config
except ImportError:
    print("Error: Could not import config. Ensure config.py is in the project root and sys.path is correct.")
    sys.exit(1)

# --- Previous functions (create_speech_specific_vad_file, remove_authors_from_vad_file) can be here or imported ---

def sample_fully_localized_speech_candidates(
    vad_pickle_filename="train_voice_data_no_known_speech_authors.pkl",
    output_sample_dirname="localized_speech_candidates",
    num_samples_to_copy=30, # Number of candidate files to copy for listening
    beginning_zone_percent=0.20,
    end_zone_percent=0.20,
    min_total_speech_duration_for_check=0.5 # Only consider files with at least this much VAD activity
):
    """
    Identifies files where ALL VAD segments are in the beginning or end zones.
    Samples these files, copies them to an output directory, and prints their VAD timestamps.
    """
    vad_pickle_path = os.path.join(config.VOICE_SEPARATION_DIR, vad_pickle_filename)
    # Output directory will be eda/voice/localized_speech_candidates relative to project root
    base_output_dir_relative_to_eda = "voice"
    full_output_dir = os.path.join(config.PROJECT_ROOT, "eda", base_output_dir_relative_to_eda, output_sample_dirname)
    
    audio_base_dir = config.train_audio_dir
    rare_audio_dir = config.train_audio_rare_dir

    print(f"Loading VAD data from: {vad_pickle_path}")
    try:
        with open(vad_pickle_path, 'rb') as f:
            vad_data = pickle.load(f)
        if not isinstance(vad_data, dict):
            print(f"Error: Expected a dictionary in {vad_pickle_path}, but found {type(vad_data)}. Exiting.")
            return
        print(f"Successfully loaded {len(vad_data)} entries from {vad_pickle_path}.")
    except FileNotFoundError:
        print(f"Error: VAD pickle file not found at {vad_pickle_path}. Exiting.")
        return
    except Exception as e:
        print(f"Error loading VAD pickle: {e}. Exiting.")
        return

    localized_candidates_info = []
    files_processed = 0
    audio_files_not_found_count = 0
    librosa_error_count = 0
    zero_duration_audio_count = 0

    print(f"Identifying files with all VAD segments in first {beginning_zone_percent*100:.0f}% or last {end_zone_percent*100:.0f}%...")
    for filename_key, intervals in tqdm(vad_data.items(), desc="Analyzing segment localization"):
        files_processed += 1
        
        if not isinstance(intervals, list) or not intervals:
            continue # Skip if no intervals for this file

        # Calculate total VAD speech for this file first
        current_file_total_vad_speech = 0.0
        for segment in intervals:
            if isinstance(segment, dict) and 'start' in segment and 'end' in segment:
                try:
                    current_file_total_vad_speech += max(0, float(segment['end']) - float(segment['start']))
                except (ValueError, TypeError):
                    continue # Skip malformed segment
        
        if current_file_total_vad_speech < min_total_speech_duration_for_check:
            continue # Skip if not enough VAD activity to warrant this specific check

        # Determine full audio file path
        audio_file_path = os.path.join(audio_base_dir, filename_key)
        if not os.path.exists(audio_file_path) and os.path.exists(rare_audio_dir):
            audio_file_path = os.path.join(rare_audio_dir, filename_key)
        elif not os.path.exists(audio_file_path):
            # Check if filename_key itself is an absolute path (legacy)
            if os.path.exists(filename_key):
                audio_file_path = filename_key
            else:
                audio_files_not_found_count += 1
                continue

        try:
            audio_total_duration_seconds = librosa.get_duration(filename=audio_file_path)
        except Exception:
            librosa_error_count += 1
            continue
        
        if audio_total_duration_seconds == 0:
            zero_duration_audio_count +=1
            continue

        beginning_zone_strict_end_time = audio_total_duration_seconds * beginning_zone_percent
        end_zone_strict_start_time = audio_total_duration_seconds * (1 - end_zone_percent)

        all_segments_fully_localized = True
        for segment in intervals:
            if not (isinstance(segment, dict) and 'start' in segment and 'end' in segment):
                all_segments_fully_localized = False # Should not happen if file had intervals
                break
            try:
                seg_start = float(segment['start'])
                seg_end = float(segment['end'])
            except (ValueError, TypeError):
                all_segments_fully_localized = False # Malformed segment, treat as not localized
                break

            # A segment is considered NOT localized if any part of it is in the middle zone.
            # This means its start must be >= end_zone_strict_start_time OR its end must be <= beginning_zone_strict_end_time.
            # If seg_end > beginning_zone_strict_end_time AND seg_start < end_zone_strict_start_time, it means some part is in middle.
            if seg_end > beginning_zone_strict_end_time and seg_start < end_zone_strict_start_time:
                all_segments_fully_localized = False
                break 
        
        if all_segments_fully_localized:
            localized_candidates_info.append({"filename": filename_key, "intervals": intervals, "audio_path": audio_file_path})

    print(f"\n--- Localization Analysis Summary ---")
    print(f"Total files processed from VAD data: {files_processed}")
    print(f"Files identified as having ALL VAD segments in outer zones: {len(localized_candidates_info)}")
    if audio_files_not_found_count > 0:
        print(f"Audio files not found during analysis: {audio_files_not_found_count}")
    if librosa_error_count > 0:
        print(f"Errors getting duration with librosa: {librosa_error_count}")
    if zero_duration_audio_count > 0:
        print(f"Audio files with zero duration: {zero_duration_audio_count}")

    if not localized_candidates_info:
        print("No files met the criteria for fully localized speech. No samples will be copied.")
        return

    num_to_actually_sample = min(len(localized_candidates_info), num_samples_to_copy)
    if num_to_actually_sample == 0:
        print("No candidates to sample after filtering.")
        return
        
    sampled_candidates = random.sample(localized_candidates_info, num_to_actually_sample)

    os.makedirs(full_output_dir, exist_ok=True)
    print(f"\nSampling {num_to_actually_sample} files and copying to: {full_output_dir}")
    print("Details of sampled files (VAD intervals will be printed below):")

    copied_count = 0
    for i, candidate_info in enumerate(sampled_candidates):
        src_audio_path = candidate_info["audio_path"]
        relative_filename = candidate_info["filename"] # This is species_folder/basename.ogg
        # For the destination, just use the basename to avoid creating species subfolders in the sample dir
        audio_basename = os.path.basename(relative_filename)
        dst_audio_path = os.path.join(full_output_dir, audio_basename)
        
        try:
            shutil.copy(src_audio_path, dst_audio_path)
            print(f"  {i+1}. Copied: {relative_filename} (Source: {src_audio_path}) -> {dst_audio_path}")
            print(f"     VAD Intervals: {candidate_info['intervals']}")
            copied_count +=1
        except Exception as e_copy:
            print(f"  Error copying {src_audio_path} to {dst_audio_path}: {e_copy}")
            print(f"     (Still printing VAD intervals for {relative_filename}: {candidate_info['intervals']})")
            
    print(f"\nFinished sampling. Copied {copied_count} audio files for review.")

if __name__ == "__main__":
    # ---- Define authors list previously used (for context, not directly used by this new function) ----
    authors_previously_handled = [
        "Mauricio Álvarez-Rebolledo (Colección de Sonidos Ambientales - Instituto Humboldt)",
        "Gary Stiles (Colección de Sonidos Ambientales - Instituto Humboldt)",
        "Mauricio Álvarez Rebolledo (Colección de Sonidos Ambientales - Instituto Humboldt)",
        "Sergio Córdoba-Córdoba (Colección de Sonidos Ambientales - Instituto Humboldt)",
        "Mauricio Álvarez-Rebolledo"
    ]

    # ---- Call the new function to sample fully localized speech candidates ----
    print("Starting script to sample audio files with VAD segments fully localized to ends...")
    sample_fully_localized_speech_candidates(
        vad_pickle_filename="train_voice_data_no_known_speech_authors.pkl", 
        output_sample_dirname="localized_speech_candidates", 
        num_samples_to_copy=30,
        beginning_zone_percent=0.20,
        end_zone_percent=0.20,
        min_total_speech_duration_for_check=0.5 
    )

    # ---- Previous script calls (commented out) ----
    # print("\n---------------------------------------\n")
    # print("Creating a VAD file specifically for segments from designated authors (assumed speech)...")
    # create_speech_specific_vad_file(
    #     source_vad_pickle_filename="train_voice_data_cleaned.pkl", 
    #     output_vad_pickle_filename="train_voice_data_final.pkl",  
    #     authors_to_include=authors_previously_handled
    # ) 
    # print("\n---------------------------------------\n")
    # print("Removing VAD segments from authors (now in 'final' speech file) from the 'cleaned' VAD file...")
    # remove_authors_from_vad_file(
    #     source_vad_pickle_filename="train_voice_data_cleaned.pkl",
    #     output_vad_pickle_filename="train_voice_data_no_known_speech_authors.pkl",
    #     authors_to_remove=authors_previously_handled
    # )
