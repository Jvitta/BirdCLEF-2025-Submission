import os
import glob
import librosa
import soundfile
import tqdm
from config import config # Import the config instance directly
import logging
import multiprocessing
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_single_file(file_path, output_dir, chunk_duration_sec, sample_rate):
    """Processes a single audio file: loads, chunks, and saves chunks."""
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=sample_rate, mono=True)

        chunk_length_samples = int(chunk_duration_sec * sample_rate)
        num_chunks = len(y) // chunk_length_samples

        if num_chunks == 0:
            # Return 0 chunks saved for this file
            return os.path.basename(file_path), 0, None # filename, chunks_saved, error

        chunks_saved_for_file = 0
        # Extract and save each chunk
        for i in range(num_chunks):
            start_sample = i * chunk_length_samples
            end_sample = start_sample + chunk_length_samples
            y_chunk = y[start_sample:end_sample]

            # Construct output filename
            original_filename_stem = os.path.splitext(os.path.basename(file_path))[0]
            output_filename = os.path.join(output_dir, f"{original_filename_stem}_chunk{i:04d}.wav")

            # Save the chunk
            soundfile.write(output_filename, y_chunk, sample_rate)
            chunks_saved_for_file += 1

        return os.path.basename(file_path), chunks_saved_for_file, None # filename, chunks_saved, error

    except Exception as e:
        # Return error for this file
        return os.path.basename(file_path), 0, str(e) # filename, chunks_saved, error

def create_chunks_parallel(input_dir, output_dir, chunk_duration_sec, sample_rate, num_workers):
    """
    Loads audio files from input_dir, splits them into chunks in parallel, and saves them to output_dir.
    """
    logging.info(f"Starting parallel chunk creation with {num_workers} workers.")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Chunk duration: {chunk_duration_sec} seconds")
    logging.info(f"Sample rate: {sample_rate} Hz")

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Ensured output directory exists: {output_dir}")

    input_files = glob.glob(os.path.join(input_dir, '*.ogg'), recursive=False)
    if not input_files:
        logging.warning(f"No .ogg files found in {input_dir}. Exiting.")
        return
    logging.info(f"Found {len(input_files)} .ogg files to process.")

    # Create a partial function with fixed arguments for the worker
    worker_func = partial(process_single_file,
                          output_dir=output_dir,
                          chunk_duration_sec=chunk_duration_sec,
                          sample_rate=sample_rate)

    processed_files = 0
    total_chunks_saved = 0
    errors = 0

    # Use multiprocessing Pool
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap_unordered for potentially better performance with I/O
        # Wrap with tqdm for progress bar
        results = tqdm.tqdm(pool.imap_unordered(worker_func, input_files), total=len(input_files), desc="Processing Soundscapes")

        for filename, chunks_saved, error_msg in results:
            if error_msg:
                logging.error(f"Error processing file {filename}: {error_msg}")
                errors += 1
            elif chunks_saved == 0:
                 logging.warning(f"File {filename} was shorter than chunk duration or resulted in 0 chunks.")
                 processed_files += 1 # Count it as processed even if no chunks saved
            else:
                total_chunks_saved += chunks_saved
                processed_files += 1

    logging.info(f"Finished processing.")
    logging.info(f"Successfully processed {processed_files - errors} out of {len(input_files)} files.")
    if errors > 0:
        logging.warning(f"Encountered errors in {errors} files.")
    logging.info(f"Saved a total of {total_chunks_saved} chunks to {output_dir}.")

if __name__ == "__main__":
    # Make sure CHUNK_DURATION_SEC is defined in config.py (e.g., CHUNK_DURATION_SEC = 5)
    # We use config.num_workers which defaults to cpu_count - 1
    create_chunks_parallel(
        input_dir=config.unlabeled_audio_dir,
        output_dir=config.unlabeled_audio_dir_chunked,
        chunk_duration_sec=5, # Ensure this exists in config.py!
        sample_rate=config.FS,
        num_workers=config.num_workers
    ) 