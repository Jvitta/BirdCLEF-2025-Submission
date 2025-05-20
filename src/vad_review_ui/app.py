import gradio as gr
import os
import pandas as pd
import pickle
from datetime import datetime
import sys
import json # For storing list of VAD segments as string
import numpy as np # Added for loading NPZ file

# --- Project Setup ---
# Assuming this app.py is in src/vad_review_ui/
# So, config.py is two levels up from current_dir then in project_root
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_from_app = os.path.dirname(os.path.dirname(current_script_dir))

if project_root_from_app not in sys.path:
    sys.path.append(project_root_from_app)

try:
    from config import config
except ImportError:
    print("Error: Could not import config.py. Ensure it's in the project root and sys.path is correct.")
    # Fallback for critical paths if config fails - adjust as necessary for your environment
    # This part is tricky because config holds essential paths.
    # For a robust app, ensure config.py is always found.
    # As a placeholder if running without proper config for some reason (not recommended for full functionality):
    class MockConfig:
        VOICE_SEPARATION_DIR = os.path.join(project_root_from_app, "data", "BC25 voice separation") # Example
        train_csv_path = os.path.join(project_root_from_app, "data", "raw", "train.csv") # Example
        train_audio_dir = os.path.join(project_root_from_app, "data", "raw", "train_audio") # Example
        train_audio_rare_dir = os.path.join(project_root_from_app, "data", "raw", "train_audio_rare") # Example
        PROJECT_ROOT = project_root_from_app
        BIRDNET_DETECTIONS_NPZ_PATH = os.path.join(PROJECT_ROOT, "outputs", "preprocessed", "birdnet_detections.npz") # Added for mock
    config = MockConfig()
    print("Warning: Using mock config paths as config.py import failed. Please check your setup.")


# --- Constants ---
VAD_PICKLE_NAME = "train_voice_data_no_known_speech_authors.pkl"
VAD_PICKLE_PATH = os.path.join(config.VOICE_SEPARATION_DIR, VAD_PICKLE_NAME)
REVIEW_LOG_CSV_NAME = "vad_review_log.csv"
REVIEW_LOG_CSV_PATH = os.path.join(current_script_dir, REVIEW_LOG_CSV_NAME) # Save log in the app's directory

# --- Helper Functions ---

def get_audio_file_path(filename_key):
    """Constructs the full path to an audio file, checking main and rare directories."""
    path_main = os.path.join(config.train_audio_dir, filename_key)
    if os.path.exists(path_main):
        return path_main
    
    path_rare = os.path.join(config.train_audio_rare_dir, filename_key)
    if hasattr(config, 'train_audio_rare_dir') and os.path.exists(config.train_audio_rare_dir) and os.path.exists(path_rare):
        return path_rare
    
    if os.path.exists(filename_key):
        return filename_key
        
    print(f"Warning: Audio file not found for key: {filename_key} in main or rare dirs.")
    return None

def load_birdnet_detections_filenames(npz_path):
    """Loads BirdNET NPZ and returns a set of filenames that have detections."""
    if not os.path.exists(npz_path):
        print(f"Info: BirdNET detections NPZ file not found at {npz_path}. No files will be excluded based on BirdNET presence.")
        return set()
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            # data.files directly gives the list of keys (filenames) in the NPZ
            filenames_with_detections = set(data.files)
        print(f"Successfully loaded BirdNET detection keys for {len(filenames_with_detections)} files from {npz_path}.")
        return filenames_with_detections
    except Exception as e:
        print(f"Error loading BirdNET detections NPZ {npz_path}: {e}. Assuming no files have BirdNET detections for filtering.")
        return set()

def load_vad_data(pickle_path):
    """Loads the VAD dictionary (filename_key: [segments])."""
    if not os.path.exists(pickle_path):
        print(f"Error: VAD pickle file not found at {pickle_path}")
        return {}
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded VAD data from {pickle_path} with {len(data)} entries.")
        return data
    except Exception as e:
        print(f"Error loading VAD pickle {pickle_path}: {e}")
        return {}

def load_train_metadata(csv_path):
    """Loads train.csv and creates a filename_key -> {author, target_common_name} mapping."""
    filename_to_metadata_map = {}
    if not os.path.exists(csv_path):
        print(f"Error: Train metadata CSV not found at {csv_path}")
        return filename_to_metadata_map # Return empty if no metadata
    try:
        df = pd.read_csv(csv_path)
        if 'filename' in df.columns and 'author' in df.columns and 'common_name' in df.columns:
            df['filename'] = df['filename'].astype(str).str.replace(r'[\\\\/]+', '/', regex=True)
            for _, row in df.iterrows():
                filename_to_metadata_map[row['filename']] = {
                    'author': row['author'],
                    'target_common_name': row['common_name']
                }
            print(f"Successfully loaded train metadata from {csv_path} and created filename_to_metadata_map for {len(filename_to_metadata_map)} files.")
        else:
            print(f"Error: 'filename', 'author', or 'common_name' column missing in {csv_path}.")
    except Exception as e:
        print(f"Error loading train metadata {csv_path}: {e}")
    return filename_to_metadata_map

def get_files_to_exclude_based_on_birdnet(raw_vad_file_keys, birdnet_npz_path, filename_to_metadata_map, debug_limit=15):
    """Identifies files from VAD list that should be excluded because BirdNET detected their target species."""
    files_to_exclude = set()
    birdnet_all_detections = {}
    debug_printed_count = 0

    if not os.path.exists(birdnet_npz_path):
        print(f"Info: BirdNET detections NPZ file not found at {birdnet_npz_path}. No files will be excluded based on BirdNET presence.")
        return files_to_exclude
    try:
        birdnet_all_detections_loaded = np.load(birdnet_npz_path, allow_pickle=True)
        if hasattr(birdnet_all_detections_loaded, 'files'):
             birdnet_all_detections = {key: birdnet_all_detections_loaded[key] for key in birdnet_all_detections_loaded.files}
        else:
             birdnet_all_detections = dict(birdnet_all_detections_loaded)
        print(f"Successfully loaded BirdNET detections for {len(birdnet_all_detections)} files from {birdnet_npz_path}.")
    except Exception as e:
        print(f"Error loading BirdNET detections NPZ {birdnet_npz_path}: {e}. Assuming no files have BirdNET detections for filtering.")
        return files_to_exclude

    print(f"\n--- Debugging BirdNET based exclusion (first ~{debug_limit} VAD files) ---")
    for filename_key in raw_vad_file_keys:
        metadata = filename_to_metadata_map.get(filename_key)
        target_common_name_for_debug = metadata.get('target_common_name', 'N/A') if metadata else 'N/A'

        if filename_key in birdnet_all_detections:
            detections_for_file = birdnet_all_detections[filename_key]
            # Check if detections_for_file is a list/array and is not empty
            if (isinstance(detections_for_file, (list, np.ndarray)) and len(detections_for_file) > 0):
                files_to_exclude.add(filename_key)
                if debug_printed_count < debug_limit:
                    print(f"  VAD file: {filename_key:<40} | Target: {target_common_name_for_debug:<25} | Status: EXCLUDED (Target species detections found in BirdNET NPZ)")
                    debug_printed_count += 1
            else:
                # File is in NPZ, but the detection list is empty (shouldn't happen if birdnet_preprocessing.py only saves non-empty)
                if debug_printed_count < debug_limit:
                    print(f"  VAD file: {filename_key:<40} | Target: {target_common_name_for_debug:<25} | Status: Kept (Present in BirdNET NPZ but with EMPTY detection list)")
                    debug_printed_count += 1
        else:
            # File is not in BirdNET NPZ at all
            if debug_printed_count < debug_limit:
                print(f"  VAD file: {filename_key:<40} | Target: {target_common_name_for_debug:<25} | Status: Kept (Not found in BirdNET NPZ keys)")
                debug_printed_count += 1
        
        if not metadata and debug_printed_count < debug_limit and filename_key not in files_to_exclude : # Add this check to print missing metadata for files that weren't excluded already
             # Check if it was already printed due to other reasons
            already_printed_status = False
            if filename_key in birdnet_all_detections:
                if not (isinstance(birdnet_all_detections[filename_key], (list, np.ndarray)) and len(birdnet_all_detections[filename_key]) > 0):
                    already_printed_status = True # Printed as "Kept (Present in BirdNET NPZ but with EMPTY detection list)"
            else:
                already_printed_status = True # Printed as "Kept (Not found in BirdNET NPZ keys)"

            if not already_printed_status:
                 print(f"  VAD file: {filename_key:<40} | Target: N/A                     | Status: Kept (No metadata in train.csv; BirdNET status handled above)")
                 debug_printed_count +=1


    print("--- End of BirdNET exclusion debugging ---")
    return files_to_exclude

def load_review_log(log_path):
    """Loads existing review decisions. Initializes an empty log if not found."""
    log_columns = ["filename_key", "species", "author", "decision", "review_timestamp", "original_vad_segments_str"]
    if os.path.exists(log_path):
        try:
            df = pd.read_csv(log_path)
            # Normalize filename_key just in case
            if 'filename_key' in df.columns:
                 df['filename_key'] = df['filename_key'].astype(str).str.replace(r'[\\\\/]+', '/', regex=True)
            else: # If critical column is missing, better to re-init
                print(f"Warning: 'filename_key' missing in {log_path}. Re-initializing log.")
                df = pd.DataFrame(columns=log_columns)

            # Ensure all columns exist, add if missing (e.g., for older log versions)
            for col in log_columns:
                if col not in df.columns:
                    df[col] = pd.NA # Or appropriate default
            
            print(f"Successfully loaded review log from {log_path} with {len(df)} entries.")
            # Ensure correct dtypes, esp for string columns that might be read as float if all NA
            df['filename_key'] = df['filename_key'].astype(str)
            df['decision'] = df['decision'].astype(str)


            reviewed_set = set(df['filename_key'].astype(str).unique())
            return df, reviewed_set
        except pd.errors.EmptyDataError:
            print(f"Review log {log_path} is empty. Initializing a new one.")
            df = pd.DataFrame(columns=log_columns)
            return df, set()
        except Exception as e:
            print(f"Error loading review log {log_path}: {e}. Initializing a new one.")
            df = pd.DataFrame(columns=log_columns)
            return df, set()
    else:
        print(f"Review log not found at {log_path}. Initializing a new one.")
        df = pd.DataFrame(columns=log_columns)
        return df, set()

def save_review_log(df, log_path):
    """Saves the review log DataFrame to CSV."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        df.to_csv(log_path, index=False)
        print(f"Review log saved to {log_path} with {len(df)} entries.")
        return True
    except Exception as e:
        print(f"Error saving review log to {log_path}: {e}")
        return False

# --- Initial Data Loading & Preparation ---
print("Starting VAD Review UI: Loading initial data...")
raw_vad_data = load_vad_data(VAD_PICKLE_PATH)
filename_to_metadata_map = load_train_metadata(config.train_csv_path)
review_log_df_global, reviewed_filenames_set_global = load_review_log(REVIEW_LOG_CSV_PATH)

# Get set of files to exclude based on BirdNET detecting the target species
files_to_exclude_set = get_files_to_exclude_based_on_birdnet(
    raw_vad_data.keys(), 
    config.BIRDNET_DETECTIONS_NPZ_PATH, 
    filename_to_metadata_map
)

# Prepare the master list of all VAD files with their info
# This list will be filtered for display.
all_vad_files_info_list = []
if raw_vad_data:
    files_in_vad_data_count = 0
    files_excluded_count = 0
    for fn_key, segments in raw_vad_data.items():
        files_in_vad_data_count += 1
        if fn_key in files_to_exclude_set:
            files_excluded_count += 1
            continue
            
        metadata = filename_to_metadata_map.get(fn_key, {})
        author = metadata.get("author", "unknown_author")
        # Use target_common_name as species for display, fallback to folder name
        species_display_name = metadata.get("target_common_name", fn_key.split('/')[0] if '/' in fn_key else "unknown_species")
        
        all_vad_files_info_list.append({
            "filename_key": fn_key,
            "species": species_display_name, # This is now common_name
            "author": author,
            "vad_segments": segments
        })
    print(f"\nProcessed {files_in_vad_data_count} files from VAD data source.")
    if files_excluded_count > 0:
        print(f"Excluded {files_excluded_count} files because their target species was detected by BirdNET.")
    print(f"Prepared master list of {len(all_vad_files_info_list)} VAD files for review (after target BirdNET exclusion).")
else:
    print("Warning: No VAD data loaded. The application will have no files to review.")

# --- Gradio UI Definition ---

def get_initial_dropdown_choices(all_info, reviewed_set):
    """Gets unique species and authors from unreviewed files."""
    unreviewed_files = [f_info for f_info in all_info if f_info["filename_key"] not in reviewed_set]
    # 'species' field in all_info now holds the common_name
    species_choices = sorted(list(set(f_info["species"] for f_info in unreviewed_files)))
    author_choices = sorted(list(set(f_info["author"] for f_info in unreviewed_files)))
    return ["All"] + species_choices, ["All"] + author_choices

initial_species_choices, initial_author_choices = get_initial_dropdown_choices(all_vad_files_info_list, reviewed_filenames_set_global)

with gr.Blocks(title="VAD Segment Review UI") as demo:
    gr.Markdown("# VAD Segment Review UI")
    gr.Markdown(f"Reviewing VAD segments from: `{VAD_PICKLE_NAME}`. Decisions saved to: `{REVIEW_LOG_CSV_NAME}`.")

    # --- State Management ---
    # Store the global loaded data in states to be accessible by UI functions
    # These are initialized once and then primarily read by UI functions.
    # Modifications to review_log_df and reviewed_set will be handled carefully.
    
    # Master list of all file information (rarely changes unless app restarts)
    s_all_vad_files_info = gr.State(value=all_vad_files_info_list)
    
    # Log of reviews (DataFrame, updates with each decision)
    s_review_log_df = gr.State(value=review_log_df_global)
    
    # Set of reviewed filenames (updates with each decision)
    s_reviewed_filenames_set = gr.State(value=reviewed_filenames_set_global)
    
    # Current queue of files to display based on filters (list of filename_keys)
    s_display_queue = gr.State(value=[]) # Will be populated by filter_and_update_queue
    
    # Index for the current file in s_display_queue
    s_current_file_idx = gr.State(value=0)
    
    # Store the current filename_key being reviewed to pass to save function
    s_current_filename_key = gr.State(value=None)
    s_current_vad_segments = gr.State(value=None) # To log original segments

    with gr.Row():
        with gr.Column(scale=1):
            species_dropdown = gr.Dropdown(label="Filter by Species (Common Name)", choices=initial_species_choices, value="All")
        with gr.Column(scale=1):
            author_dropdown = gr.Dropdown(label="Filter by Author", choices=initial_author_choices, value="All")
        with gr.Column(scale=1, min_width=150):
            apply_filters_button = gr.Button("Apply Filters / Refresh Queue")
    
    with gr.Row():
        current_file_display = gr.Markdown("Current File: N/A")
    
    progress_display = gr.Markdown("Progress: N/A")

    audio_player = gr.Audio(label="Audio Player", type="filepath", interactive=False)
    vad_segments_display = gr.Textbox(label="VAD Segments (Time: Start - End)", lines=5, interactive=False)

    with gr.Row():
        keep_button = gr.Button("✅ Keep All Segments (Good)")
        discard_button = gr.Button("❌ Discard All Segments (Bad)")
        mixed_quality_button = gr.Button("⚠️ Mixed Quality (Needs Manual Edit)")
        # unsure_button = gr.Button("❓ Mark as Unsure / Review Later") # Optional
    
    next_file_button = gr.Button("➡️ Next Unreviewed File in Queue")
    status_message_display = gr.Markdown("")

    # --- UI Logic Functions ---
    
    def format_vad_segments_for_display(segments_list):
        if not segments_list or not isinstance(segments_list, list):
            return "No VAD segments for this file."
        display_str = ""
        for i, seg in enumerate(segments_list):
            try:
                start_s = float(seg['start'])
                end_s = float(seg['end'])
                
                start_minutes = int(start_s // 60)
                start_seconds_part = start_s % 60
                
                end_minutes = int(end_s // 60)
                end_seconds_part = end_s % 60
                
                # Format seconds with leading zero if needed, and two decimal places
                start_formatted_seconds = f"{start_seconds_part:05.2f}" 
                end_formatted_seconds = f"{end_seconds_part:05.2f}"

                display_str += f"{i+1}: {start_minutes}:{start_formatted_seconds} - {end_minutes}:{end_formatted_seconds}\\n"
            except (TypeError, KeyError, ValueError) as e:
                display_str += f"{i+1}: Error parsing segment data {seg} - {e}\\n"
        return display_str.strip()

    def load_file_for_display(filename_key_to_load, all_vad_info_list_state):
        """Loads a single file's data into the UI components."""
        if not filename_key_to_load:
            return (
                "N/A (No file selected or queue empty)", 
                None, # audio_player path
                "N/A", # vad_segments_display
                None, # s_current_filename_key update
                None  # s_current_vad_segments update
            )

        file_info = next((item for item in all_vad_info_list_state if item["filename_key"] == filename_key_to_load), None)

        if not file_info:
            return (
                f"Error: File info not found for {filename_key_to_load}", 
                None, "Error", 
                filename_key_to_load, # still update current key for context
                None 
            )
        
        audio_path = get_audio_file_path(file_info["filename_key"])
        vad_display = format_vad_segments_for_display(file_info["vad_segments"])
        
        author_name = file_info.get("author", "Unknown Author")
        # species field now contains common name
        file_display_md = f"**Current File:** `{file_info['filename_key']}` (Species: `{file_info['species']}`, Author: `{author_name}`)"
        
        return (
            file_display_md, 
            audio_path, 
            vad_display,
            file_info["filename_key"], # Update s_current_filename_key
            file_info["vad_segments"]  # Update s_current_vad_segments
        )

    def filter_and_update_queue_and_ui(sel_species, sel_author, 
                                    all_vad_info_list_state, reviewed_set_state,
                                    # These are outputs to other components:
                                    # species_dropdown_choices, author_dropdown_choices,
                                    # progress_display_text
                                    ):
        """
        Filters the master list based on selections, updates the display queue,
        and loads the first file from the new queue.
        Also updates dropdown choices based on the new unreviewed set.
        """
        
        # 1. Filter all_vad_info_list_state
        filtered_for_queue = []
        for f_info in all_vad_info_list_state:
            if f_info["filename_key"] in reviewed_set_state:
                continue # Skip already reviewed

            species_match = (sel_species == "All" or f_info["species"] == sel_species)
            author_match = (sel_author == "All" or f_info["author"] == sel_author)
            
            if species_match and author_match:
                filtered_for_queue.append(f_info["filename_key"])
        
        new_display_queue = sorted(list(set(filtered_for_queue))) # Sort for consistent order
        new_current_idx = 0
        
        # 2. Update dropdown choices based on *all* unreviewed files globally
        #    This gives users visibility into what's left overall, not just in current filter
        current_unreviewed_files = [f for f in all_vad_info_list_state if f["filename_key"] not in reviewed_set_state]
        
        # Species choices from remaining unreviewed files that match current author filter (if not 'All')
        species_choices_for_dropdown = ["All"] + sorted(list(set(
            f_info["species"] for f_info in current_unreviewed_files 
            if (sel_author == "All" or f_info["author"] == sel_author)
        )))
        
        # Author choices from remaining unreviewed files that match current species filter (if not 'All')
        author_choices_for_dropdown = ["All"] + sorted(list(set(
            f_info["author"] for f_info in current_unreviewed_files
            if (sel_species == "All" or f_info["species"] == sel_species)
        )))

        # 3. Load the first file from the new_display_queue
        next_filename_to_load = new_display_queue[0] if new_display_queue else None
        
        file_display_md, audio_path, vad_display, \
        current_fn_key_update, current_vad_seg_update = load_file_for_display(next_filename_to_load, all_vad_info_list_state)

        # 4. Update progress display
        total_unreviewed_overall = len(current_unreviewed_files)
        queue_len = len(new_display_queue)
        progress_text = f"Queue: {queue_len} file(s) matching filters. File {new_current_idx + 1 if queue_len > 0 else 0} of {queue_len}. (Total unreviewed: {total_unreviewed_overall})"

        return (
            new_display_queue, new_current_idx, # Updates to s_display_queue, s_current_file_idx
            gr.update(choices=species_choices_for_dropdown, value=sel_species), # Update species dropdown
            gr.update(choices=author_choices_for_dropdown, value=sel_author), # Update author dropdown
            progress_text, # Update progress_display
            file_display_md, audio_path, vad_display, # Updates for file display elements
            current_fn_key_update, current_vad_seg_update # Update current file states
        )

    def handle_next_file_click(current_idx_state, display_queue_state, all_vad_info_list_state, reviewed_set_state, sel_species, sel_author):
        """Handles the 'Next File' button click."""
        new_idx = current_idx_state + 1
        
        if new_idx < len(display_queue_state):
            next_filename_to_load = display_queue_state[new_idx]
            file_display_md, audio_path, vad_display, \
            current_fn_key_update, current_vad_seg_update = load_file_for_display(next_filename_to_load, all_vad_info_list_state)
            
            # Update progress display
            total_unreviewed_overall = len([f for f in all_vad_info_list_state if f["filename_key"] not in reviewed_set_state])
            queue_len = len(display_queue_state)
            progress_text = f"Queue: {queue_len} file(s) matching filters. File {new_idx + 1} of {queue_len}. (Total unreviewed: {total_unreviewed_overall})"
            
            return (
                new_idx, # Update s_current_file_idx
                progress_text,
                file_display_md, audio_path, vad_display,
                current_fn_key_update, current_vad_seg_update
            )
        else:
            # At the end of the queue
            gr.Info("Reached the end of the current review queue for these filters!")
            # Optionally, could auto-refresh the queue here if desired, or just indicate end.
            # For now, just update progress and indicate end.
            total_unreviewed_overall = len([f for f in all_vad_info_list_state if f["filename_key"] not in reviewed_set_state])
            queue_len = len(display_queue_state)
            progress_text = f"Queue: End Reached ({queue_len} file(s)). (Total unreviewed: {total_unreviewed_overall})"

            return (
                current_idx_state, # No change to index
                progress_text,
                "End of Queue. Apply new filters or restart.", None, "N/A",
                None, None # Clear current file states
            )


    def process_decision(decision_type, 
                        current_fn_key_state, current_vad_segments_state, # From s_current_filename_key, s_current_vad_segments
                        review_log_df_state, reviewed_set_state, # From s_review_log_df, s_reviewed_filenames_set
                        # For reloading next:
                        current_idx_state, display_queue_state, all_vad_info_list_state,
                        sel_species_state, sel_author_state # For re-filtering/maintaining context
                        ):
        if not current_fn_key_state:
            return "No file selected to make a decision.", review_log_df_state, reviewed_set_state, \
                   current_idx_state, display_queue_state, \
                   gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update() # No changes if no file, ensure all outputs match

        # 1. Log the decision
        filename_key = current_fn_key_state
        file_info = next((f for f in all_vad_info_list_state if f["filename_key"] == filename_key), {})
        species = file_info.get("species", "unknown_species") # species is now common_name
        author = file_info.get("author", "unknown_author")
        
        # Convert VAD segments to JSON string for CSV storage
        vad_segments_str = json.dumps(current_vad_segments_state) if current_vad_segments_state else "[]"

        new_log_entry = pd.DataFrame([{
            "filename_key": filename_key,
            "species": species,
            "author": author,
            "decision": decision_type,
            "review_timestamp": datetime.now().isoformat(),
            "original_vad_segments_str": vad_segments_str
        }])
        
        # Remove any previous decision for this file before adding the new one
        updated_review_log_df = review_log_df_state[review_log_df_state['filename_key'] != filename_key]
        updated_review_log_df = pd.concat([updated_review_log_df, new_log_entry], ignore_index=True)
        
        save_success = save_review_log(updated_review_log_df, REVIEW_LOG_CSV_PATH)
        status_msg = f"Decision '{decision_type}' for {filename_key} {'saved.' if save_success else 'save FAILED.'}"
        
        updated_reviewed_set = reviewed_set_state.copy()
        updated_reviewed_set.add(filename_key)

        # 2. Refresh the queue (excluding the just reviewed file) and load the next file from the *current* queue position
        #    The `filter_and_update_queue_and_ui` function inherently handles skipping reviewed files.
        #    We call it again with the same filters to refresh the queue and automatically get the next
        #    unreviewed item at the current effective index 0 of the *new* queue.
        
        (new_display_queue, new_current_idx, 
        species_dd_update, author_dd_update,
        progress_text_update, 
        file_display_md_update, audio_path_update, vad_display_update,
        next_fn_key_update, next_vad_seg_update) = filter_and_update_queue_and_ui(
                                                        sel_species_state, sel_author_state,
                                                        all_vad_info_list_state, updated_reviewed_set
                                                    )
        
        if not new_display_queue: # If queue becomes empty after this decision
            status_msg += " All files in current filter are now reviewed."
            file_display_md_update = "All files in current filter reviewed. Apply new filters."
            audio_path_update = None
            vad_display_update = "N/A"
            next_fn_key_update = None
            next_vad_seg_update = None
        
        return (
            status_msg, 
            updated_review_log_df, # Update s_review_log_df
            updated_reviewed_set,  # Update s_reviewed_filenames_set
            new_display_queue,     # Update s_display_queue
            new_current_idx,       # Update s_current_file_idx
            # UI updates:
            species_dd_update, 
            author_dd_update,
            progress_text_update,
            file_display_md_update,
            audio_path_update,
            vad_display_update,
            next_fn_key_update, # Update s_current_filename_key
            next_vad_seg_update # Update s_current_vad_segments
        )

    # --- Connect UI Components to Logic ---
    
    # Initial load when app starts / UI is ready
    # We use .then() to ensure state is available.
    # The `filter_and_update_queue_and_ui` acts as the initial loader.
    demo.load( # Use demo.load for initial population based on default dropdown values
        fn=filter_and_update_queue_and_ui,
        inputs=[
            species_dropdown, author_dropdown, # Current values of dropdowns ("All", "All")
            s_all_vad_files_info, s_reviewed_filenames_set # Global states
        ],
        outputs=[
            s_display_queue, s_current_file_idx, # Update queue and index states
            species_dropdown, author_dropdown,    # Update dropdown choices themselves
            progress_display,                     # Update progress text
            current_file_display, audio_player, vad_segments_display, # Update file display
            s_current_filename_key, s_current_vad_segments # Update current file context states
        ]
    )
    
    apply_filters_button.click(
        fn=filter_and_update_queue_and_ui,
        inputs=[
            species_dropdown, author_dropdown, 
            s_all_vad_files_info, s_reviewed_filenames_set
        ],
        outputs=[
            s_display_queue, s_current_file_idx,
            species_dropdown, author_dropdown, 
            progress_display,
            current_file_display, audio_player, vad_segments_display,
            s_current_filename_key, s_current_vad_segments
        ]
    )

    next_file_button.click(
        fn=handle_next_file_click,
        inputs=[
            s_current_file_idx, s_display_queue, 
            s_all_vad_files_info, s_reviewed_filenames_set,
            species_dropdown, author_dropdown # Pass current filter values for context in progress text
        ],
        outputs=[
            s_current_file_idx, # Update state
            progress_display,
            current_file_display, audio_player, vad_segments_display,
            s_current_filename_key, s_current_vad_segments
        ]
    )

    keep_button.click(
        fn=process_decision,
        inputs=[
            gr.State(value="kept"), # Decision type
            s_current_filename_key, s_current_vad_segments,
            s_review_log_df, s_reviewed_filenames_set,
            s_current_file_idx, s_display_queue, s_all_vad_files_info, # For reloading next
            species_dropdown, author_dropdown # For maintaining filter context when reloading
        ],
        outputs=[
            status_message_display,
            s_review_log_df, s_reviewed_filenames_set, # Update states
            s_display_queue, s_current_file_idx,      # Update queue states
            # UI component updates:
            species_dropdown, author_dropdown,
            progress_display,
            current_file_display, audio_player, vad_segments_display,
            s_current_filename_key, s_current_vad_segments # Update current file context states
        ]
    )
    
    discard_button.click(
        fn=process_decision,
        inputs=[
            gr.State(value="discarded"), # Decision type
            s_current_filename_key, s_current_vad_segments,
            s_review_log_df, s_reviewed_filenames_set,
            s_current_file_idx, s_display_queue, s_all_vad_files_info,
            species_dropdown, author_dropdown
        ],
        outputs=[
            status_message_display,
            s_review_log_df, s_reviewed_filenames_set,
            s_display_queue, s_current_file_idx,
            species_dropdown, author_dropdown,
            progress_display,
            current_file_display, audio_player, vad_segments_display,
            s_current_filename_key, s_current_vad_segments
        ]
    )

    mixed_quality_button.click(
        fn=process_decision,
        inputs=[
            gr.State(value="mixed_quality"), # Decision type
            s_current_filename_key, s_current_vad_segments,
            s_review_log_df, s_reviewed_filenames_set,
            s_current_file_idx, s_display_queue, s_all_vad_files_info,
            species_dropdown, author_dropdown
        ],
        outputs=[
            status_message_display,
            s_review_log_df, s_reviewed_filenames_set, # Update states
            s_display_queue, s_current_file_idx,      # Update queue states
            # UI component updates:
            species_dropdown, author_dropdown,
            progress_display,
            current_file_display, audio_player, vad_segments_display,
            s_current_filename_key, s_current_vad_segments # Update current file context states
        ]
    )

# --- Launch the UI ---
if __name__ == "__main__":
    if not all_vad_files_info_list:
        print("CRITICAL WARNING: all_vad_files_info_list is empty. VAD data might not have loaded correctly or is empty. UI will be limited.")
    
    print(f"Review log is at: {REVIEW_LOG_CSV_PATH}")
    print(f"VAD data source: {VAD_PICKLE_PATH}")
    print(f"Number of initial species choices (incl. All): {len(initial_species_choices)}")
    print(f"Number of initial author choices (incl. All): {len(initial_author_choices)}")
    print(f"Total files in VAD data prepared for UI (after exclusions): {len(all_vad_files_info_list)}")
    print(f"Total already reviewed files according to log: {len(reviewed_filenames_set_global)}")

    demo.launch(share=False, debug=True)


