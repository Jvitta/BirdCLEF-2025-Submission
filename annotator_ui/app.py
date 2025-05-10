import gradio as gr
import os
import pandas as pd
from datetime import datetime
import sys

PROJECT_ROOT_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT_APP not in sys.path:
    sys.path.append(PROJECT_ROOT_APP)

from config import config

OUTPUT_ANNOTATIONS_FILE = os.path.join(config.PROJECT_ROOT, "annotator_ui", "annotated_segments.csv")

# --- Helper Functions ---
def load_audio_structure():
    """Scans the train_audio directory and organizes files by species folder."""
    audio_structure = {}
    try:
        for species_folder_name in sorted(os.listdir(config.train_audio_dir)):
            species_folder_path = os.path.join(config.train_audio_dir, species_folder_name)
            if os.path.isdir(species_folder_path):
                audio_files_in_folder = []
                for file_basename in sorted(os.listdir(species_folder_path)):
                    if file_basename.lower().endswith(('.ogg', '.mp3', '.wav', '.flac', '.m4a')):
                        audio_files_in_folder.append(file_basename)
                if audio_files_in_folder: # Only add folder if it has audio files
                    audio_structure[species_folder_name] = audio_files_in_folder
        return audio_structure
    except Exception as e:
        print(f"Error loading audio structure: {e}")
        return {}

def get_primary_label(filename_relative_to_train_audio, train_df):
    """Gets primary label from the main training metadata.
       filename_relative_to_train_audio should be like 'species_folder/actual_file.ogg'
    """
    if train_df is None or filename_relative_to_train_audio not in train_df['filename'].values:
        return "unknown"
    return train_df[train_df['filename'] == filename_relative_to_train_audio]['primary_label'].iloc[0]

# --- Load initial data ---
audio_data_structure = load_audio_structure()
species_folders_list = list(audio_data_structure.keys())

try:
    main_train_df = pd.read_csv(config.train_csv_path)
except Exception as e:
    print(f"Warning: Could not load main train metadata {config.train_csv_path}: {e}")
    main_train_df = None

# --- Annotation Data Handling ---
# Define columns for the new annotation structure
ANNOTATION_COLUMNS = ["filename", "center_time_s", "primary_label", "annotation_time"]

if os.path.exists(OUTPUT_ANNOTATIONS_FILE):
    try:
        annotations_df = pd.read_csv(OUTPUT_ANNOTATIONS_FILE)
        if not all(col in annotations_df.columns for col in ANNOTATION_COLUMNS): # Check if columns match
            print(f"Warning: Annotation file columns mismatch. Re-initializing {OUTPUT_ANNOTATIONS_FILE}.")
            annotations_df = pd.DataFrame(columns=ANNOTATION_COLUMNS)
    except pd.errors.EmptyDataError:
        annotations_df = pd.DataFrame(columns=ANNOTATION_COLUMNS)
    except Exception as e:
        print(f"Error loading existing annotations: {e}. Re-initializing.")
        annotations_df = pd.DataFrame(columns=ANNOTATION_COLUMNS)
else:
    annotations_df = pd.DataFrame(columns=ANNOTATION_COLUMNS)

def save_annotation(selected_species_folder, selected_audio_basename, center_time_str):
    global annotations_df
    if not selected_species_folder or not selected_audio_basename:
        return "Please select a species folder and an audio file first.", annotations_df_to_display()
    if not center_time_str:
        return "Please set the center timestamp using the button.", annotations_df_to_display()
        
    try:
        center_s = float(center_time_str)

        if center_s < 0:
            return "Timestamp must be non-negative.", annotations_df_to_display()

        filename_relative = os.path.join(selected_species_folder, selected_audio_basename)
        primary_label = get_primary_label(filename_relative, main_train_df)

        new_annotation = pd.DataFrame([{\
            "filename": filename_relative,
            "center_time_s": center_s,
            "primary_label": primary_label,
            "annotation_time": datetime.now().isoformat()
        }])
        
        annotations_df = pd.concat([annotations_df, new_annotation], ignore_index=True)
        
        # Ensure the output directory exists before saving the CSV
        output_dir = os.path.dirname(OUTPUT_ANNOTATIONS_FILE)
        os.makedirs(output_dir, exist_ok=True)
        
        annotations_df.to_csv(OUTPUT_ANNOTATIONS_FILE, index=False)
        
        return f"Annotation saved for {filename_relative} at {center_s:.2f}s.", annotations_df_to_display()
    except ValueError:
        return "Invalid timestamp format in center time field. Please ensure it's a number.", annotations_df_to_display()
    except Exception as e:
        return f"Error saving annotation: {str(e)}", annotations_df_to_display()

def annotations_df_to_display():
    return annotations_df.tail().to_html(index=False) if not annotations_df.empty else "No annotations yet."

# --- Gradio UI Definition ---
with gr.Blocks(title="Bird Call Segment Annotator") as demo:
    gr.Markdown("# Bird Call Segment Annotator")
    gr.Markdown(
        "Select a species folder, then an audio file. Play it. "
        "Use the 'Set Center Time' button to capture the current playback time for the center of a 5s segment. "
        "Then click 'Save Annotation'."
    )

    with gr.Row():
        species_folder_dropdown = gr.Dropdown(choices=species_folders_list, label="Select Species Folder")
        audio_file_dropdown = gr.Dropdown(choices=[], label="Select Audio File From Folder", interactive=True)
    
    audio_player = gr.Audio(label="Audio Player", type="filepath", interactive=False, elem_id="audio_player_element")
    
    center_time_js_output = gr.Textbox(label="Segment Center Time (JS)", interactive=True, elem_id="center_time_js_output_elem")

    def update_audio_file_choices(selected_species_folder):
        if selected_species_folder and selected_species_folder in audio_data_structure:
            files_in_folder = audio_data_structure[selected_species_folder]
            return gr.update(choices=files_in_folder, value=None, interactive=True), None, "" # Clear time textbox
        return gr.update(choices=[], value=None, interactive=True), None, "" # Clear time textbox

    species_folder_dropdown.change(
        fn=update_audio_file_choices, 
        inputs=species_folder_dropdown, 
        outputs=[audio_file_dropdown, audio_player, center_time_js_output] # Updated outputs
    )

    def update_audio_player_on_file_select(sel_species_folder, sel_audio_basename):
        if sel_species_folder and sel_audio_basename:
            relative_path = os.path.join(sel_species_folder, sel_audio_basename)
            full_path = os.path.join(config.train_audio_dir, relative_path)
            return full_path, "" # Clear time textbox
        return None, "" # Clear time textbox

    audio_file_dropdown.change(
        fn=update_audio_player_on_file_select, 
        inputs=[species_folder_dropdown, audio_file_dropdown], 
        outputs=[audio_player, center_time_js_output] # Updated outputs
    )
    
    gr.Markdown("---")
    
    # Updated JavaScript function to read from <time id="time">
    js_set_time_func = """
    (audio_player_wrapper_id_str, textbox_id_str) => {
        const audioPlayerWrapper = document.getElementById(audio_player_wrapper_id_str);
        const textboxContainer = document.getElementById(textbox_id_str);
        let timeInSeconds = null;

        if (!audioPlayerWrapper) {
            console.error('JS Error: Audio player WRAPPER element not found with ID:', audio_player_wrapper_id_str);
            if (textboxContainer) {
                const errInputElement = textboxContainer.querySelector('textarea') || textboxContainer.querySelector('input[type="text"]');
                if (errInputElement) errInputElement.value = "Error: AudioW NotFound";
            }
            return null;
        }

        const timeDisplayElement = audioPlayerWrapper.querySelector('div[data-testid="waveform-Audio Player"] #time'); // More specific selector for Gradio's <time id="time">

        if (!timeDisplayElement) {
            console.error('JS Error: Time display element (#time) not found within wrapper:', audio_player_wrapper_id_str, 'Attempted selector:', 'div[data-testid="waveform-Audio Player"] #time');
             // Fallback attempt for older Gradio versions or different structures if the above fails
            const simplerTimeDisplayElement = audioPlayerWrapper.querySelector('#time');
            if (!simplerTimeDisplayElement) {
                console.error('JS Error: Simpler time display element (#time) also not found within wrapper:', audio_player_wrapper_id_str);
                 if (textboxContainer) {
                    const errInputElement = textboxContainer.querySelector('textarea') || textboxContainer.querySelector('input[type="text"]');
                    if (errInputElement) errInputElement.value = "Error: TimeElm NotFound";
                }
                return null;
            }
            console.log('JS Info: Found time element with simpler selector #time');
            // If found with simpler selector, use it. This is a common structure.
            // The original code did not reassign timeDisplayElement here, fixed.
            // No, stick to one attempt or make it clearer. The first selector is more robust if it works.
            // Let's assume the structure seen in the image and try to make that robust.
            // The image shows <div id="audio_player_element"> containing <div class="component-wrapper" data-testid="waveform-Audio Player"> which then contains <time id="time">
            // So, a more direct approach relative to audio_player_wrapper_id_str (which is "audio_player_element")
            // The provided screenshot implies audio_player_element *IS* the div that contains component-wrapper.
        }
        
        // Re-evaluating selector based on the image:
        // `audio_player_element` is the ID of a div.
        // Inside it, there's a structure leading to `<div class="timestamps svelte-19usgod"> <time id="time" ...>`
        // A more direct query from the wrapper `audio_player_element` might be:
        const actualTimeElement = audioPlayerWrapper.querySelector('#time'); // Standard <time> element with id="time"

        if (!actualTimeElement) {
             console.error('JS Error: Final attempt to find #time element failed within wrapper:', audio_player_wrapper_id_str);
            if (textboxContainer) {
                const errInputElement = textboxContainer.querySelector('textarea') || textboxContainer.querySelector('input[type="text"]');
                if (errInputElement) errInputElement.value = "Error: TimeElm NF";
            }
            return null;
        }


        const timeString = actualTimeElement.textContent; // e.g., "0:03", "1:25"
        console.log('JS Info: Raw timeString from #time element:', timeString);

        if (typeof timeString === 'string' && timeString.includes(':')) {
            const parts = timeString.split(':');
            if (parts.length === 2) {
                const minutes = parseInt(parts[0], 10);
                const seconds = parseInt(parts[1], 10);
                if (!isNaN(minutes) && !isNaN(seconds)) {
                    timeInSeconds = (minutes * 60) + seconds;
                } else {
                    console.error('JS Error: Could not parse minutes/seconds from timeString:', timeString);
                }
            } else if (parts.length === 3) { // For HH:MM:SS
                const hours = parseInt(parts[0], 10);
                const minutes = parseInt(parts[1], 10);
                const seconds = parseInt(parts[2], 10);
                if (!isNaN(hours) && !isNaN(minutes) && !isNaN(seconds)) {
                    timeInSeconds = (hours * 3600) + (minutes * 60) + seconds;
                } else {
                    console.error('JS Error: Could not parse hours/minutes/seconds from timeString:', timeString);
                }
            }
             else {
                console.error('JS Error: timeString does not have 2 or 3 parts (unexpected format):', timeString);
            }
        } else {
            console.error('JS Error: timeString is not valid or does not contain ":" :', timeString);
        }

        if (timeInSeconds === null) {
            if (textboxContainer) {
                const errInputElement = textboxContainer.querySelector('textarea') || textboxContainer.querySelector('input[type="text"]');
                if (errInputElement) errInputElement.value = "Error: TimeParseFail";
            }
            return null;
        }
        
        console.log('JS Info: Parsed timeInSeconds:', timeInSeconds);

        if (textboxContainer) {
            const inputElement = textboxContainer.querySelector('textarea') || textboxContainer.querySelector('input[type="text"]');
            if (inputElement) {
                inputElement.value = timeInSeconds.toFixed(3); // Store as float string, e.g., "3.000" or "85.000"
                const event = new Event('input', { bubbles: true });
                inputElement.dispatchEvent(event);
                const changeEvent = new Event('change', { bubbles: true });
                inputElement.dispatchEvent(changeEvent);
                console.log('JS Info: Successfully set textbox', textbox_id_str, 'to', timeInSeconds.toFixed(3));
            } else {
                console.error('JS Error: Could not find the input/textarea field within textbox ID:', textbox_id_str);
            }
        } else {
            console.error('JS Error: Textbox container not found with ID:', textbox_id_str);
        }
        return timeInSeconds;
    }
    """
    
    set_center_time_button = gr.Button("Set Center Time")
        
    set_center_time_button.click(
        fn=None, 
        inputs=None, outputs=None, 
        # Pass the ID of the audio player's main div and the target textbox's div
        js=f"( () => {{ const fn = {js_set_time_func}; fn('audio_player_element', 'center_time_js_output_elem'); }} )()"
    )
        
    save_button = gr.Button("Save Annotation")
    
    status_message = gr.Markdown()
    gr.Markdown("### Recent Annotations:")
    annotations_table = gr.HTML(value=annotations_df_to_display())

    save_button.click(
        fn=save_annotation,
        inputs=[species_folder_dropdown, audio_file_dropdown, center_time_js_output], # Updated inputs
        outputs=[status_message, annotations_table]
    )
    save_button.click(lambda: "", outputs=center_time_js_output) # Clear center time textbox

# --- Launch the UI ---
if __name__ == "__main__":
    print(f"Audio folders found: {len(species_folders_list)}")
    if not species_folders_list:
        print("Warning: No audio folders found in train_audio_dir. The first dropdown will be empty.")
    print(f"Existing annotations loaded: {len(annotations_df)} rows.")
    print(f"Annotations will be saved to: {OUTPUT_ANNOTATIONS_FILE}")
    demo.launch(share=False)