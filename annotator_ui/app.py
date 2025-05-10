import gradio as gr
import os
import pandas as pd
from datetime import datetime
import sys
import numpy as np

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

# Load taxonomy data
taxonomy_df = None
try:
    taxonomy_df = pd.read_csv(config.taxonomy_path)
    if 'primary_label' in taxonomy_df.columns:
        taxonomy_df['primary_label'] = taxonomy_df['primary_label'].astype(str)
    else:
        print(f"Warning: 'primary_label' column not found in {config.taxonomy_path}")
        taxonomy_df = None
except Exception as e:
    print(f"Error loading taxonomy data from {config.taxonomy_path}: {e}")

# Load BirdNET detections
all_birdnet_detections = {}
try:
    with np.load(config.BIRDNET_DETECTIONS_NPZ_PATH, allow_pickle=True) as data:
        all_birdnet_detections = {key: data[key] for key in data.files}
    print(f"Successfully loaded BirdNET detections for {len(all_birdnet_detections)} files.")
except FileNotFoundError:
    print(f"Warning: BirdNET detections file not found at {config.BIRDNET_DETECTIONS_NPZ_PATH}. Proceeding without BirdNET filtering.")
except Exception as e:
    print(f"Warning: Error loading BirdNET detections NPZ from {config.BIRDNET_DETECTIONS_NPZ_PATH}: {e}. Proceeding without BirdNET filtering.")

# --- Annotation Data Handling ---
# Define columns for the new annotation structure
ANNOTATION_COLUMNS = ["filename", "center_time_s", "primary_label", "annotation_time"]

if os.path.exists(OUTPUT_ANNOTATIONS_FILE):
    try:
        annotations_df = pd.read_csv(OUTPUT_ANNOTATIONS_FILE)
        if not all(col in annotations_df.columns for col in ANNOTATION_COLUMNS): # Check if columns match
            print(f"Warning: Annotation file columns mismatch. Re-initializing {OUTPUT_ANNOTATIONS_FILE}.")
            annotations_df = pd.DataFrame(columns=ANNOTATION_COLUMNS)
        # Ensure filename column is string type for consistent matching
        if 'filename' in annotations_df.columns:
            annotations_df['filename'] = annotations_df['filename'].astype(str)
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
        with gr.Column(scale=2):
            species_folder_dropdown = gr.Dropdown(choices=species_folders_list, label="Select Species Folder")
            audio_file_dropdown_no_bn = gr.Dropdown(choices=[], label="Select Audio File (No BirdNET Detections)", interactive=True)
            audio_file_dropdown_with_bn = gr.Dropdown(choices=[], label="Select Audio File (Has BirdNET Detections)", interactive=True)
        with gr.Column(scale=1):
            gr.Markdown("### Species Information")
            scientific_name_md = gr.Markdown("**Scientific Name:** N/A")
            common_name_md = gr.Markdown("**Common Name:** N/A")
            class_name_md = gr.Markdown("**Class:** N/A")
            birdnet_detections_md = gr.Markdown("#### Top BirdNET Detections:\nN/A")
    
    audio_player = gr.Audio(label="Audio Player", type="filepath", interactive=False, elem_id="audio_player_element")
    
    center_time_js_output = gr.Textbox(label="Segment Center Time (JS)", interactive=True, elem_id="center_time_js_output_elem")

    def update_audio_file_choices_and_species_info(selected_species_folder):
        # Defaults for species info
        sci_name_text = "**Scientific Name:** N/A"
        com_name_text = "**Common Name:** N/A"
        cls_name_text = "**Class:** N/A"
        # Initialize updates for two dropdowns
        audio_choices_no_bn_update = gr.update(choices=[], value=None, interactive=True)
        audio_choices_with_bn_update = gr.update(choices=[], value=None, interactive=True)
        audio_player_update = None
        center_time_update = ""
        birdnet_detections_text = "#### Top BirdNET Detections:\nN/A"

        if selected_species_folder and selected_species_folder in audio_data_structure:
            all_files_in_folder = audio_data_structure[selected_species_folder]
            
            annotated_files_in_current_folder_set = set()
            if not annotations_df.empty and 'filename' in annotations_df.columns:
                prefix_to_check = selected_species_folder + os.path.sep
                relevant_annotations = annotations_df[annotations_df['filename'].str.startswith(prefix_to_check, na=False)]
                annotated_files_in_current_folder_set = set(relevant_annotations['filename'])
            
            files_to_consider_for_dropdowns = []
            for file_basename in all_files_in_folder:
                relative_path = os.path.join(selected_species_folder, file_basename)
                if relative_path in annotated_files_in_current_folder_set:
                    continue 
                files_to_consider_for_dropdowns.append(file_basename)
            
            choices_no_bn = []
            choices_with_bn_intermediate = [] # Store (basename, max_confidence) for sorting

            for file_basename in files_to_consider_for_dropdowns:
                relative_path_os_specific = os.path.join(selected_species_folder, file_basename)
                lookup_key_for_birdnet = relative_path_os_specific.replace(os.path.sep, '/')
                
                has_any_birdnet_detection = False
                max_confidence_for_file = 0.0 # Default for sorting if no valid dets

                if lookup_key_for_birdnet in all_birdnet_detections:
                    bn_dets_raw = all_birdnet_detections[lookup_key_for_birdnet]
                    if bn_dets_raw is not None: 
                        # Process bn_dets_raw to a list of detection dicts to find max confidence
                        actual_detections = []
                        if isinstance(bn_dets_raw, np.ndarray):
                            actual_detections = [item for item in bn_dets_raw if isinstance(item, dict)]
                        elif isinstance(bn_dets_raw, list):
                            actual_detections = [d for d in bn_dets_raw if isinstance(d, dict)]
                        
                        if actual_detections: # If we have a list of dicts
                            has_any_birdnet_detection = True
                            confidences = [det.get('confidence', 0.0) for det in actual_detections if isinstance(det.get('confidence'), (int, float))]
                            if confidences:
                                max_confidence_for_file = max(confidences)
                
                if has_any_birdnet_detection:
                    choices_with_bn_intermediate.append((file_basename, max_confidence_for_file))
                else:
                    choices_no_bn.append(file_basename) 
            
            # Sort choices_with_bn_intermediate by max_confidence (descending)
            choices_with_bn_intermediate.sort(key=lambda x: x[1], reverse=True)
            # Extract just the basenames for the dropdown
            final_choices_with_bn = [basename for basename, conf in choices_with_bn_intermediate]

            audio_choices_no_bn_update = gr.update(choices=sorted(choices_no_bn), value=None, interactive=True)
            audio_choices_with_bn_update = gr.update(choices=final_choices_with_bn, value=None, interactive=True)
            
            if taxonomy_df is not None and 'primary_label' in taxonomy_df.columns:
                species_info = taxonomy_df[taxonomy_df['primary_label'] == str(selected_species_folder)]
                if not species_info.empty:
                    sci_name = species_info['scientific_name'].iloc[0] if 'scientific_name' in species_info else "N/A"
                    com_name = species_info['common_name'].iloc[0] if 'common_name' in species_info else "N/A"
                    cls_name = species_info['class_name'].iloc[0] if 'class_name' in species_info else "N/A"
                    
                    sci_name_text = f"**Scientific Name:** {sci_name}"
                    com_name_text = f"**Common Name:** {com_name}"
                    cls_name_text = f"**Class:** {cls_name}"
                else:
                    print(f"Info: Species folder '{selected_species_folder}' not found in taxonomy_df.")
            else:
                print("Info: taxonomy_df is not loaded or 'primary_label' column is missing.")
        
        # Return updates for both dropdowns and the new markdown
        return (
            audio_choices_no_bn_update, audio_choices_with_bn_update, 
            audio_player_update, center_time_update, 
            sci_name_text, com_name_text, cls_name_text, 
            birdnet_detections_text
        )

    species_folder_dropdown.change(
        fn=update_audio_file_choices_and_species_info,
        inputs=species_folder_dropdown, 
        outputs=[
            audio_file_dropdown_no_bn, audio_file_dropdown_with_bn, 
            audio_player, center_time_js_output, 
            scientific_name_md, common_name_md, class_name_md,
            birdnet_detections_md
        ]
    )

    def update_audio_player_and_bn_info(sel_species_folder, sel_audio_basename):
        audio_player_path = None
        center_time_val = ""
        bn_info_text = "#### Top BirdNET Detections:\nN/A"

        if sel_species_folder and sel_audio_basename:
            relative_path = os.path.join(sel_species_folder, sel_audio_basename)
            full_path = os.path.join(config.train_audio_dir, relative_path)
            audio_player_path = full_path
            center_time_val = "" # Clear time textbox

            # Fetch and format BirdNET detections
            lookup_key_for_birdnet = relative_path.replace(os.path.sep, '/')

            if lookup_key_for_birdnet in all_birdnet_detections:
                bn_dets = all_birdnet_detections[lookup_key_for_birdnet]

                if bn_dets is not None and ((isinstance(bn_dets, np.ndarray) and bn_dets.size > 0) or (isinstance(bn_dets, list) and len(bn_dets) > 0)):
                    detections_to_sort = []
                    if isinstance(bn_dets, np.ndarray):
                        # If bn_dets is a numpy array, assume its elements are the dicts
                        # Convert to list of dicts, filtering for actual dicts just in case
                        detections_to_sort = [item for item in bn_dets if isinstance(item, dict)]
                    elif isinstance(bn_dets, list):
                        # If it's already a list, filter for dicts
                        detections_to_sort = [d for d in bn_dets if isinstance(d, dict)]
                    
                    if detections_to_sort: # Check if we have a non-empty list of dictionaries
                        valid_dets = sorted(detections_to_sort, key=lambda x: x.get('confidence', 0), reverse=True)
                        
                        bn_info_text = "#### Top BirdNET Detections (Top 5):\n"
                        for i, det in enumerate(valid_dets[:5]):
                            start = det.get('start_time', 'N/A')
                            end = det.get('end_time', 'N/A')
                            conf = det.get('confidence', 0)
                            # Ensure start/end are numbers before formatting, otherwise use 'N/A'
                            start_str = f"{start:.1f}s" if isinstance(start, (int, float)) else str(start)
                            end_str = f"{end:.1f}s" if isinstance(end, (int, float)) else str(end)
                            conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
                            bn_info_text += f"{i+1}. Start: {start_str}, End: {end_str}, Conf: {conf_str}\n"
                    else:
                        bn_info_text = "#### Top BirdNET Detections:\nNo valid dictionary-formatted detections found."
                else:
                    bn_info_text = "#### Top BirdNET Detections:\nNo detections listed for this file (data is None or empty)."
            else:
                bn_info_text = "#### Top BirdNET Detections:\nFile not found in BirdNET detection records."

        return audio_player_path, center_time_val, bn_info_text

    # Event handler for the 'No BirdNET' dropdown
    audio_file_dropdown_no_bn.change(
        fn=update_audio_player_and_bn_info, # Use new function
        inputs=[species_folder_dropdown, audio_file_dropdown_no_bn], 
        outputs=[audio_player, center_time_js_output, birdnet_detections_md] # Added birdnet_detections_md
    ).then(
        fn=lambda: gr.update(value=None), # Clear the other dropdown
        inputs=None,
        outputs=[audio_file_dropdown_with_bn]
    )

    # Event handler for the 'With BirdNET' dropdown
    audio_file_dropdown_with_bn.change(
        fn=update_audio_player_and_bn_info, # Use new function
        inputs=[species_folder_dropdown, audio_file_dropdown_with_bn], 
        outputs=[audio_player, center_time_js_output, birdnet_detections_md] # Added birdnet_detections_md
    ).then(
        fn=lambda: gr.update(value=None), # Clear the other dropdown
        inputs=None,
        outputs=[audio_file_dropdown_no_bn]
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

    # Modify save_annotation to pick the active dropdown
    def save_annotation_revised(selected_species_folder, audio_file_no_bn, audio_file_with_bn, center_time_str):
        global annotations_df
        
        selected_audio_basename = None
        if audio_file_no_bn:
            selected_audio_basename = audio_file_no_bn
        elif audio_file_with_bn:
            selected_audio_basename = audio_file_with_bn
        else:
            return "Please select an audio file from one of the dropdowns.", annotations_df_to_display()

        if not selected_species_folder : # sel_audio_basename is already checked
            return "Please select a species folder first.", annotations_df_to_display()
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
            
            output_dir = os.path.dirname(OUTPUT_ANNOTATIONS_FILE)
            os.makedirs(output_dir, exist_ok=True)
            
            annotations_df.to_csv(OUTPUT_ANNOTATIONS_FILE, index=False)
            
            # Also, after saving, we might want to clear both audio dropdowns
            # This requires returning updates for them.
            # For now, just return status and table. We can enhance clear later.
            return f"Annotation saved for {filename_relative} at {center_s:.2f}s.", annotations_df_to_display()
        except ValueError:
            return "Invalid timestamp format in center time field. Please ensure it's a number.", annotations_df_to_display()
        except Exception as e:
            return f"Error saving annotation: {str(e)}", annotations_df_to_display()


    save_button.click(
        fn=save_annotation_revised, # Use new save function
        inputs=[species_folder_dropdown, audio_file_dropdown_no_bn, audio_file_dropdown_with_bn, center_time_js_output], 
        outputs=[status_message, annotations_table]
    )
    # Clear only center time textbox after save, keep dropdowns selected
    save_button.click(
        fn=lambda: "", 
        inputs=None, 
        outputs=[center_time_js_output]
    )

# --- Launch the UI ---
# Remove old save_annotation if it's still there globally (it's now save_annotation_revised)
if 'save_annotation' in globals() and callable(globals()['save_annotation']):
    del globals()['save_annotation']

if __name__ == "__main__":
    print(f"Audio folders found: {len(species_folders_list)}")
    if not species_folders_list:
        print("Warning: No audio folders found in train_audio_dir. The first dropdown will be empty.")
    print(f"Existing annotations loaded: {len(annotations_df)} rows.")
    print(f"Annotations will be saved to: {OUTPUT_ANNOTATIONS_FILE}")
    demo.launch(share=False)