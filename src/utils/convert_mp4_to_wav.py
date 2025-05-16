import os
import subprocess
import sys
import argparse
import json # Added for parsing ffprobe JSON output

# Add project root to sys.path to allow importing config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from config import config as project_config # Use an alias to avoid conflict
except ImportError:
    print("Error: Could not import project_config. Make sure this script is runnable and config.py exists.")
    sys.exit(1)

def convert_audio_to_wav(input_dir, output_dir, target_sample_rate, delete_original=False):
    """
    Converts MP4, MP3 files in input_dir to WAV format in output_dir.
    Also inspects existing WAV/WAV files and re-encodes them if they are not
    mono or at the target_sample_rate. Ensures all output WAVs are lowercase .wav.
    Output directory structure mirrors the input directory structure.

    Args:
        input_dir (str): Directory containing audio files (can have subdirectories).
        output_dir (str): Directory to save WAV files (will mirror subdirectories).
        target_sample_rate (int): The target sample rate for the WAV files.
        delete_original (bool): If True, delete the original MP4/MP3/WAV after conversion/processing.
    """
    print(f"Scanning for .mp4, .mp3, .wav, .WAV files in: {input_dir} (and its subdirectories)")
    print(f"Output directory for .wav files: {output_dir} (subdirectories will be mirrored)")
    print(f"Target format: Mono, {target_sample_rate} Hz, .wav extension")
    if delete_original:
        print("Original files will be DELETED after successful processing if specified and applicable.")

    converted_count = 0
    inspected_skipped_count = 0
    error_count = 0
    renamed_count = 0

    for root, _, files in os.walk(input_dir):
        for filename in files:
            original_filepath = os.path.join(root, filename)
            original_extension = os.path.splitext(filename)[1].lower() # Work with lowercase extension

            if original_extension not in [".mp4", ".mp3", ".wav"]: # Note: .wav covers .WAV due to lower()
                continue

            # Determine corresponding output subdirectory and output wav_filepath
            relative_subdir = os.path.relpath(root, input_dir)
            current_output_dir = os.path.join(output_dir, relative_subdir)
            if relative_subdir == '.': # Handle files directly in input_dir
                current_output_dir = output_dir
            
            os.makedirs(current_output_dir, exist_ok=True)

            # Ensure output filename is lowercase .wav
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            wav_filepath = os.path.join(current_output_dir, wav_filename)

            needs_ffmpeg_processing = False
            
            if original_extension in [".mp4", ".mp3"]:
                print(f"  Processing (convert from {original_extension}): {original_filepath} -> {wav_filepath}")
                needs_ffmpeg_processing = True
            elif original_extension == ".wav": # This covers .wav and .WAV
                print(f"  Inspecting existing WAV file: {original_filepath}")
                try:
                    probe_command = [
                        'ffprobe',
                        '-v', 'quiet',
                        '-print_format', 'json',
                        '-show_streams',
                        original_filepath
                    ]
                    probe_process = subprocess.run(probe_command, capture_output=True, text=True, check=False)
                    
                    if probe_process.returncode == 0 and probe_process.stdout:
                        stream_info = json.loads(probe_process.stdout)
                        audio_streams = [s for s in stream_info.get('streams', []) if s.get('codec_type') == 'audio']
                        
                        if not audio_streams:
                            print(f"    Warning: No audio stream found in {original_filepath}. Assuming conversion needed.")
                            needs_ffmpeg_processing = True
                        else:
                            # Use the first audio stream
                            current_channels = audio_streams[0].get('channels')
                            current_sr_str = audio_streams[0].get('sample_rate')
                            current_sr = int(current_sr_str) if current_sr_str else -1

                            if current_channels == 1 and current_sr == target_sample_rate:
                                inspected_skipped_count += 1
                                # Check if a rename is needed (e.g. .WAV to .wav, or if output_dir is different but file is compliant)
                                if original_filepath != wav_filepath and os.path.abspath(original_filepath) != os.path.abspath(wav_filepath):
                                    # This can happen if input is .WAV or if input/output dirs are different.
                                    # If output_dir is same as input_dir, this handles .WAV -> .wav rename.
                                    # If output_dir is different, this effectively copies the compliant file with lowercase .wav.
                                    print(f"    Standardizing path/extension: {original_filepath} -> {wav_filepath}")
                                    try:
                                        if os.path.abspath(original_filepath) == os.path.abspath(wav_filepath): # Case change only
                                             os.rename(original_filepath, wav_filepath)
                                             renamed_count+=1
                                        else: # Different paths or output_dir scenario, copy the file
                                             subprocess.run(['cp', original_filepath, wav_filepath], check=True)
                                             renamed_count+=1 # Consider this a form of "conversion" to standard path
                                        
                                        if delete_original and original_filepath != wav_filepath and os.path.exists(original_filepath):
                                            # Only delete if the source is different from target and flag is set
                                            # Useful if copying to a new location and original .WAV should be removed.
                                            # Avoids deleting if it was just a case rename in the same location.
                                            if os.path.abspath(original_filepath) != os.path.abspath(wav_filepath):
                                                os.remove(original_filepath)
                                                print(f"    Deleted original after standardizing path: {original_filepath}")

                                    except Exception as e_mv_cp:
                                        print(f"    Error standardizing path for {original_filepath} to {wav_filepath}: {e_mv_cp}")
                                        error_count += 1
                                continue # Skip ffmpeg processing for this compliant file
                            else:
                                print(f"    Needs re-encoding: Channels={current_channels} (target=1), SR={current_sr} (target={target_sample_rate} Hz)")
                                needs_ffmpeg_processing = True
                    else:
                        print(f"    Warning: ffprobe failed or gave no output for {original_filepath}. Assuming conversion needed. stderr: {probe_process.stderr}")
                        needs_ffmpeg_processing = True
                except FileNotFoundError:
                    print("CRITICAL Error: ffprobe (ffmpeg) command not found. Please ensure ffmpeg is installed and in your PATH.")
                    sys.exit(1)
                except json.JSONDecodeError:
                    print(f"    Warning: Could not parse ffprobe JSON output for {original_filepath}. Assuming conversion needed.")
                    needs_ffmpeg_processing = True
                except Exception as e_probe:
                    print(f"    An unexpected error occurred during ffprobe for {original_filepath}: {e_probe}. Assuming conversion needed.")
                    needs_ffmpeg_processing = True
            
            if needs_ffmpeg_processing:
                
                is_in_place_wav_conversion = (original_extension == ".wav" and os.path.abspath(original_filepath) == os.path.abspath(wav_filepath))
                temp_wav_filepath = ""

                if is_in_place_wav_conversion:
                    # Create a temporary filename for in-place WAV conversion
                    temp_wav_filename = os.path.splitext(filename)[0] + "_temp" + ".wav"
                    temp_wav_filepath = os.path.join(current_output_dir, temp_wav_filename)
                    ffmpeg_output_target = temp_wav_filepath
                    print(f"  Processing via ffmpeg (in-place): {original_filepath} -> {temp_wav_filepath} (temp) -> {wav_filepath}")
                else:
                    ffmpeg_output_target = wav_filepath
                    print(f"  Processing via ffmpeg: {original_filepath} -> {ffmpeg_output_target}")

                try:
                    command = [
                        'ffmpeg',
                        '-i', original_filepath,
                        '-vn', 
                        '-acodec', 'pcm_s16le',
                        '-ar', str(target_sample_rate),
                        '-ac', '1',
                        '-y', 
                        ffmpeg_output_target # Use temp file if applicable
                    ]
                    
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate()

                    if process.returncode == 0:
                        print(f"    Successfully processed to: {ffmpeg_output_target}")
                        
                        if is_in_place_wav_conversion:
                            try:
                                os.remove(original_filepath) # Delete original non-compliant WAV
                                os.rename(temp_wav_filepath, wav_filepath) # Rename temp to original name
                                print(f"    Successfully replaced original with standardized version: {wav_filepath}")
                                converted_count += 1
                            except Exception as e_replace:
                                print(f"    Error replacing original file {original_filepath} with temp file {temp_wav_filepath}: {e_replace}")
                                error_count += 1
                                # Attempt to clean up temp file if rename failed
                                if os.path.exists(temp_wav_filepath):
                                    try:
                                        os.remove(temp_wav_filepath)
                                    except OSError:
                                        pass # Silently ignore if temp removal fails
                        else: # MP3/MP4 conversion, or WAV to different output dir
                            converted_count += 1
                            # Delete original if flag is set AND it's not the same file path
                            if delete_original and os.path.abspath(original_filepath) != os.path.abspath(wav_filepath) and os.path.exists(original_filepath):
                                try:
                                    os.remove(original_filepath)
                                    print(f"    Deleted original: {original_filepath}")
                                except OSError as e_del:
                                    print(f"    Error deleting original {original_filepath}: {e_del}")
                    else:
                        print(f"    Error processing {original_filepath} with ffmpeg:")
                        print(f"      ffmpeg stdout: {stdout.decode('utf-8', 'ignore')}")
                        print(f"      ffmpeg stderr: {stderr.decode('utf-8', 'ignore')}")
                        error_count += 1
                        # Clean up temp file if ffmpeg failed during in-place conversion attempt
                        if is_in_place_wav_conversion and os.path.exists(temp_wav_filepath):
                            try:
                                os.remove(temp_wav_filepath)
                                print(f"    Cleaned up temporary file: {temp_wav_filepath}")
                            except OSError:
                                pass # Silently ignore if temp removal fails
                
                except FileNotFoundError: 
                    print("CRITICAL Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
                    sys.exit(1)
                except Exception as e_ffmpeg:
                    print(f"    An unexpected error occurred with ffmpeg for {original_filepath}: {e_ffmpeg}")
                    error_count += 1
                    # Clean up temp file if any other exception occurred during in-place conversion attempt
                    if is_in_place_wav_conversion and os.path.exists(temp_wav_filepath):
                        try:
                            os.remove(temp_wav_filepath)
                            print(f"    Cleaned up temporary file on error: {temp_wav_filepath}")
                        except OSError:
                            pass
                
    print(f"\nConversion and Standardization Summary:")
    print(f"  Successfully processed/converted via ffmpeg: {converted_count} files.")
    print(f"  Inspected and skipped (already compliant, no re-encoding): {inspected_skipped_count} files.")
    print(f"  Standardized path/extension (e.g. .WAV to .wav, or copied compliant to output_dir): {renamed_count} files.")
    print(f"  Errors encountered: {error_count} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MP4/MP3 audio to WAV format, and standardize existing WAV files (mono, target sample rate).")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=project_config.train_audio_rare_dir,
        help="Directory containing the audio files (.mp4, .mp3, .wav, .WAV)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=project_config.train_audio_rare_dir, 
        help="Directory to save the standardized .wav files. Mirrors input structure."
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=project_config.FS,
        help="Target sample rate for the output WAV files."
    )
    parser.add_argument(
        "--delete_original",
        action="store_true",
        help="Delete the original file after successful conversion or standardization if the output path is different (e.g., for .mp3, .mp4, or .WAV source files)."
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        sys.exit(1)

    # Renamed function call
    convert_audio_to_wav(args.input_dir, args.output_dir, args.sample_rate, args.delete_original) 