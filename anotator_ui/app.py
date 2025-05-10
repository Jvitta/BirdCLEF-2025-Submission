import gradio as gr
import os
import pandas as pd
from datetime import datetime

# --- Configuration ---
# Assuming your config.py is accessible via sys.path if running from project root
# or you adjust paths accordingly. For simplicity here, we might hardcode or use relative paths.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # ../
TRAIN_AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "train_audio")
METADATA_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "train.csv") # To get primary_label
OUTPUT_ANNOTATIONS_FILE = os.path.join(PROJECT_ROOT, "annotator_ui", "annotated_segments.csv")

# --- Helper Functions ---
def load_audio_filenames():
    """Scans the train_audio directory for audio files."""
    # This could be more sophisticated, e.g., reading from your train.csv
    # to only list files relevant to your training set.
    # For now, let's assume direct scan and filter later if needed.
    try:
        all_files = []
        # Example: Iterate through subdirectories like in train.csv (e.g., 'abethrp1/XC12345.ogg')
        for root, _, files in os.walk(TRAIN_AUDIO_DIR):
            for file in files:
                if file.lower().endswith(('.ogg', '.mp3', '.wav', '.flac', '.m4a')): # Add more extensions if needed
                    # Store relative path from TRAIN_AUDIO_DIR
                    relative_path = os.path.relpath(os.path.join(root, file), TRAIN_AUDIO_DIR)
                    all_files.append(relative_path)
        return sorted(all_files)
    except Exception as e:
        print(f"Error loading audio filenames: {e}")
        return []

def get_primary_label(filename, train_df):
    """Gets primary label from the main training metadata."""
    if train_df is None or filename not in train_df['filename'].values:
        return "unknown"
    return train_df[train_df['filename'] == filename]['primary_label'].iloc[0]

# --- Load initial data ---
audio_files_list = load_audio_filenames()
try:
    main_train_df = pd.read_csv(METADATA_CSV_PATH)
except Exception as e:
    print(f"Warning: Could not load main train metadata {METADATA_CSV_PATH}: {e}")
    main_train_df = None

# --- Annotation Function ---
# This function will be called when the user interacts with the UI
# Gradio's Audio component can output a tuple (sample_rate, numpy_array)
# but for just marking a point, we might not need to process the audio data directly here.
# We'd ideally get the click timestamp from an interactive audio player.
# Gradio's basic Audio component is for upload/playback, not precise click timestamps.
# A more advanced solution might involve a custom HTML/JS component or a different library.

# For a SIMPLER approach without direct click capture on waveform:
# User listens, notes time, enters it.
# OR, we can use the 'PlayableAudio' component if it provides better interaction for this.

# Let's make a stateful way to store annotations for the current session
if os.path.exists(OUTPUT_ANNOTATIONS_FILE):
    try:
        annotations_df = pd.read_csv(OUTPUT_ANNOTATIONS_FILE)
    except pd.errors.EmptyDataError:
        annotations_df = pd.DataFrame(columns=["filename", "center_timestamp_s", "primary_label", "annotation_time"])
    except Exception as e:
        print(f"Error loading existing annotations: {e}")
        annotations_df = pd.DataFrame(columns=["filename", "center_timestamp_s", "primary_label", "annotation_time"])

else:
    annotations_df = pd.DataFrame(columns=["filename", "center_timestamp_s", "primary_label", "annotation_time"])

def save_annotation(selected_filename, marked_timestamp_str):
    global annotations_df
    try:
        center_s = float(marked_timestamp_str)
        if center_s < 0:
            return "Timestamp must be non-negative.", annotations_df_to_display()

        primary_label = get_primary_label(selected_filename, main_train_df)

        new_annotation = pd.DataFrame([{
            "filename": selected_filename,
            "center_timestamp_s": center_s,
            "primary_label": primary_label,
            "annotation_time": datetime.now().isoformat()
        }])
        
        annotations_df = pd.concat([annotations_df, new_annotation], ignore_index=True)
        annotations_df.to_csv(OUTPUT_ANNOTATIONS_FILE, index=False)
        
        return f"Annotation saved for {selected_filename} at {center_s:.2f}s.", annotations_df_to_display()
    except ValueError:
        return "Invalid timestamp format. Please enter a number (e.g., 10.5).", annotations_df_to_display()
    except Exception as e:
        return f"Error saving annotation: {str(e)}", annotations_df_to_display()

def annotations_df_to_display():
    # Display last 5 annotations or so
    return annotations_df.tail().to_html(index=False) if not annotations_df.empty else "No annotations yet."

# --- Gradio UI Definition ---
with gr.Blocks(title="Bird Call Segment Annotator") as demo:
    gr.Markdown("# Bird Call Segment Annotator")
    gr.Markdown(
        "Select an audio file, play it, and if you hear a clear bird call, "
        "enter the approximate **center timestamp** (in seconds) of the call."
    )

    with gr.Row():
        audio_dropdown = gr.Dropdown(choices=audio_files_list, label="Select Audio File")
        # Using gr.Audio for playback. User will manually note the time.
        # The `type="filepath"` means the component expects a path to an audio file.
        audio_player = gr.Audio(label="Audio Player", type="filepath", interactive=False)

    def update_audio_player(selected_filename_relative):
        if selected_filename_relative:
            full_path = os.path.join(TRAIN_AUDIO_DIR, selected_filename_relative)
            return full_path
        return None # No audio to play if nothing selected

    audio_dropdown.change(fn=update_audio_player, inputs=audio_dropdown, outputs=audio_player)
    
    gr.Markdown("---")
    
    with gr.Row():
        timestamp_input = gr.Textbox(label="Center Timestamp of Bird Call (seconds)", placeholder="e.g., 15.7")
        # The filename for saving will be taken from the audio_dropdown's current value
        
    save_button = gr.Button("Save Annotation for Selected Audio")
    
    status_message = gr.Markdown()
    gr.Markdown("### Recent Annotations:")
    annotations_table = gr.HTML(value=annotations_df_to_display())

    save_button.click(
        fn=save_annotation,
        inputs=[audio_dropdown, timestamp_input], # Pass current value of dropdown
        outputs=[status_message, annotations_table]
    )

    # Interface to clear current timestamp input after saving (optional)
    # save_button.click(lambda: "", inputs=[], outputs=[timestamp_input])


# --- Launch the UI ---
if __name__ == "__main__":
    print(f"Audio files found: {len(audio_files_list)}")
    if not audio_files_list:
        print("Warning: No audio files found in TRAIN_AUDIO_DIR. The dropdown will be empty.")
    print(f"Existing annotations loaded: {len(annotations_df)} rows.")
    print(f"Annotations will be saved to: {OUTPUT_ANNOTATIONS_FILE}")
    demo.launch(share=False) # share=True if you want a public link (requires internet)