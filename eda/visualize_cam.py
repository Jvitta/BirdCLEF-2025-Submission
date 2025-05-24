import os
import argparse
import random
import warnings
import logging
import sys
from glob import glob # Added for finding default checkpoint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# import timm # Removed
import cv2
import matplotlib.pyplot as plt
# import torchvision.transforms as transforms # Removed for now, EfficientAT might not need ImageNet norm

# Assuming 'config.py' and 'birdclef_training.py' are accessible
from config import config
# from train import BirdCLEFModel # Use the model definition from training # Removed
from models.efficient_at.mn.model import get_model as get_efficient_at_model # Added

# Suppress warnings and limit logging output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# --- CAM Helper Functions ---

# Global variables to store hook outputs
activations = None
# gradients = None # Not needed for basic CAM, but good for Grad-CAM # Kept commented

def forward_hook(module, input, output):
    """Hook to capture the output feature maps of a layer."""
    global activations
    activations = output

# --- Main Visualization Function ---

def visualize_cam(args):
    """Loads model, data, computes CAM, and saves visualization."""
    print(f"--- Starting CAM Visualization ---")
    print(f"Using device: {args.device}")

    # --- Determine checkpoint path ---
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        print(f"No checkpoint specified, searching in: {config.MODEL_OUTPUT_DIR}")
        pth_files = sorted(glob(os.path.join(config.MODEL_OUTPUT_DIR, "*.pth")))
        if not pth_files:
            print(f"Error: No .pth files found in {config.MODEL_OUTPUT_DIR}. Please specify a checkpoint.")
            sys.exit(1)
        checkpoint_path = pth_files[0] # Use the first one found (e.g., oldest or alphabetically first)
        print(f"Using default checkpoint: {checkpoint_path}")
    
    spectrogram_npz_path = config.PREPROCESSED_NPZ_PATH # Hardcoded
    output_image_path = os.path.join(config.OUTPUT_DIR, "cam_visualization.png") # Hardcoded
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)


    print(f"Model Checkpoint: {checkpoint_path}")
    print(f"Spectrogram NPZ: {spectrogram_npz_path}")
    
    # Samplename and target class logic will be handled after loading NPZ

    print(f"Output Path: {output_image_path}")

    # --- 1. Load Taxonomy ---
    print("Loading taxonomy...")
    try:
        taxonomy_df = pd.read_csv(config.taxonomy_path)
        species_ids = taxonomy_df['primary_label'].tolist()
        idx_to_label = {idx: label for idx, label in enumerate(species_ids)}
        label_to_idx = {label: idx for idx, label in enumerate(species_ids)}
        num_classes_from_taxonomy = len(species_ids)
        if num_classes_from_taxonomy != config.num_classes:
             print(f"Warning: Taxonomy ({num_classes_from_taxonomy}) vs config ({config.num_classes}). Using taxonomy size: {num_classes_from_taxonomy}.")
        # Use num_classes_from_taxonomy for model loading if different from config
    except Exception as e:
        print(f"Error loading taxonomy: {e}")
        sys.exit(1)

    # --- 2. Load Model ---
    print("Loading model...")
    try:
        model = get_efficient_at_model(
            num_classes=num_classes_from_taxonomy,
            pretrained_name=None, 
            width_mult=config.width_mult, 
            head_type="mlp", 
            input_dim_f=config.TARGET_SHAPE[0], 
            input_dim_t=config.TARGET_SHAPE[1]
        )
        checkpoint_data = torch.load(checkpoint_path, map_location=torch.device(args.device))
        if 'model_state_dict' in checkpoint_data:
            model.load_state_dict(checkpoint_data['model_state_dict'])
        else:
             model.load_state_dict(checkpoint_data)
        model = model.to(args.device)
        model.eval()
        print(f"Model based on '{config.model_name}' (EfficientAT architecture) loaded successfully.")
    except Exception as e:
        print(f"Error loading model checkpoint {checkpoint_path}: {e}")
        sys.exit(1)

    # --- 3. Identify Target Layer and Register Hook ---
    target_layer = None
    if hasattr(model, 'features') and isinstance(model.features, nn.Sequential) and len(model.features) > 0:
        if hasattr(model.features[-1], '0') and isinstance(model.features[-1][0], nn.Conv2d):
            target_layer = model.features[-1][0] 
            print(f"Hooking into layer: model.features[-1][0] (type: {type(target_layer).__name__})")
        else:
            target_layer = model.features[-1]
            print(f"Warning: Could not find Conv2d in model.features[-1][0]. Hooking into model.features[-1] (type: {type(target_layer).__name__}). CAM might be less precise.")
    else:
        print("ERROR: Could not find 'model.features'. Inspect the model structure manually.")
        sys.exit(1)

    if target_layer:
        hook_handle = target_layer.register_forward_hook(forward_hook)
    else:
        print("ERROR: target_layer is None. Cannot register hook.")
        sys.exit(1)

    # --- 4. Load Spectrogram and Determine Samplename ---
    print("Loading spectrogram data...")
    actual_samplename_to_process = None
    try:
        with np.load(spectrogram_npz_path) as data_archive:
            all_keys = list(data_archive.keys())
            if not all_keys:
                print(f"Error: No samples found in {spectrogram_npz_path}")
                sys.exit(1)

            if args.samplename:
                if args.samplename not in data_archive:
                    print(f"Error: Specified samplename '{args.samplename}' not found in {spectrogram_npz_path}")
                    available_keys_sample = list(data_archive.keys())[:10]
                    print(f"Available keys (first 10): {available_keys_sample}")
                    sys.exit(1)
                actual_samplename_to_process = args.samplename
                spec_data = data_archive[actual_samplename_to_process]
                print(f"Using specified samplename: {actual_samplename_to_process}")
            else:
                actual_samplename_to_process = random.choice(all_keys)
                spec_data = data_archive[actual_samplename_to_process]
                print(f"No samplename provided, using random sample: {actual_samplename_to_process}")
        
        # Handle multi-chunk vs single-chunk
        if spec_data.ndim == 3:
            print(f"Input data from NPZ has shape {spec_data.shape} (multiple chunks). Selecting first chunk.")
            spec_np_5s = spec_data[0]
        elif spec_data.ndim == 2:
            spec_np_5s = spec_data
        else:
            print(f"Error: Unexpected spectrogram shape {spec_data.shape} from NPZ. Expected 2 or 3 dimensions.")
            sys.exit(1)
            
        if spec_np_5s.shape != tuple(config.PREPROCESS_TARGET_SHAPE):
             print(f"Warning: Loaded 5s spectrogram shape {spec_np_5s.shape} doesn't match config.PREPROCESS_TARGET_SHAPE {config.PREPROCESS_TARGET_SHAPE}. Resizing...")
             spec_np_5s = cv2.resize(spec_np_5s, 
                                   (config.PREPROCESS_TARGET_SHAPE[1], config.PREPROCESS_TARGET_SHAPE[0]), 
                                   interpolation=cv2.INTER_LINEAR)

        spec_np_5s_float = spec_np_5s.astype(np.float32)
        original_spec_display = spec_np_5s_float.copy() 
        spec_for_model_10s = np.concatenate([spec_np_5s_float, spec_np_5s_float], axis=1) 
        
        if spec_for_model_10s.shape != tuple(config.TARGET_SHAPE):
            print(f"Warning: Concatenated 10s spec shape {spec_for_model_10s.shape} differs from config.TARGET_SHAPE {config.TARGET_SHAPE}. Resizing to TARGET_SHAPE.")
            spec_for_model_10s = cv2.resize(spec_for_model_10s, 
                                        (config.TARGET_SHAPE[1], config.TARGET_SHAPE[0]), 
                                        interpolation=cv2.INTER_LINEAR)

        input_tensor = torch.tensor(spec_for_model_10s, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(args.device)
        print(f"Spectrogram preprocessed. Original 5s shape: {original_spec_display.shape}. Tensor shape for model (10s): {input_tensor.shape}")

    except Exception as e:
        print(f"Error loading or preprocessing spectrogram: {e}")
        sys.exit(1)

    # --- 5. Forward Pass and Get Predictions ---
    print("Performing forward pass...")
    with torch.no_grad():
        model_output = model(input_tensor)
        logits = model_output[0] if isinstance(model_output, tuple) else model_output
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    predicted_idx = np.argmax(probabilities)
    predicted_label = idx_to_label.get(predicted_idx, "Unknown")
    predicted_prob = probabilities[predicted_idx]
    print(f"Prediction: '{predicted_label}' (Index: {predicted_idx}) with probability {predicted_prob:.4f}")

    # Determine target class index for CAM
    target_idx = None
    target_label_for_cam = "Unknown"

    if args.target_class: # User-specified target_class takes highest precedence
        if args.target_class in label_to_idx:
            target_idx = label_to_idx[args.target_class]
            target_label_for_cam = args.target_class
            print(f"Using specified target class for CAM: '{target_label_for_cam}' (Index: {target_idx})")
        else:
            print(f"Error: Specified target class '{args.target_class}' not found in taxonomy. Will use predicted class instead.")
            target_idx = predicted_idx # Fallback to predicted
            target_label_for_cam = predicted_label
    
    if target_idx is None and actual_samplename_to_process: # Try to derive from samplename if not user-specified
        try:
            derived_target_class = actual_samplename_to_process.split('-')[0]
            if derived_target_class in label_to_idx:
                target_idx = label_to_idx[derived_target_class]
                target_label_for_cam = derived_target_class
                print(f"Derived target class for CAM from samplename '{actual_samplename_to_process}': '{target_label_for_cam}' (Index: {target_idx})")
            else:
                print(f"Warning: Derived class '{derived_target_class}' from samplename not in taxonomy. Will use predicted class for CAM.")
                target_idx = predicted_idx # Fallback
                target_label_for_cam = predicted_label
        except Exception:
            print(f"Warning: Could not derive target class from samplename '{actual_samplename_to_process}'. Will use predicted class for CAM.")
            target_idx = predicted_idx # Fallback
            target_label_for_cam = predicted_label

    if target_idx is None: # Fallback to predicted if all else fails
        target_idx = predicted_idx
        target_label_for_cam = predicted_label
        print(f"Using predicted class as target for CAM: '{target_label_for_cam}' (Index: {target_idx})")


    # --- 6. Calculate CAM ---
    print("Calculating CAM...")
    if activations is None:
        print("Error: Hook did not capture activations. Cannot compute CAM.")
        hook_handle.remove()
        sys.exit(1)

    try:
        if isinstance(model.classifier, nn.Sequential) and len(model.classifier) > 5 and isinstance(model.classifier[5], nn.Linear):
            classifier_weights = model.classifier[5].weight.data 
        else:
            print("Error: Could not find model.classifier[5] as nn.Linear. Inspect model structure.")
            hook_handle.remove()
            sys.exit(1)
            
        target_weights = classifier_weights[target_idx, :] 
        feature_maps = activations.squeeze(0) 
        C_feat, H_feat, W_feat = feature_maps.shape
        feature_maps_reshaped = feature_maps.view(C_feat, H_feat * W_feat)
        cam_tensor = torch.matmul(target_weights.unsqueeze(0), feature_maps_reshaped)
        cam_tensor = cam_tensor.view(H_feat, W_feat) 
        cam_tensor = F.relu(cam_tensor)

        if torch.max(cam_tensor) > 1e-6: 
            cam_tensor = cam_tensor - torch.min(cam_tensor)
            cam_tensor = cam_tensor / torch.max(cam_tensor)
        else:
            cam_tensor = torch.zeros_like(cam_tensor) 

        cam = cam_tensor.cpu().numpy() 
        original_height, original_width = original_spec_display.shape 
        cam_resized = cv2.resize(cam, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        print(f"CAM calculated. Resized to overlay on 5s spec: {cam_resized.shape}")

    except Exception as e:
        print(f"Error during CAM calculation: {e}")
        hook_handle.remove() 
        sys.exit(1)
    finally:
         hook_handle.remove() 

    # --- 7. Visualize and Save ---
    print("Generating visualization...")
    try:
        plt.style.use('default') 
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        im = ax.imshow(original_spec_display, aspect='auto', origin='lower', cmap='magma') 
        fig.colorbar(im, ax=ax, label='Amplitude/Power')
        heatmap = ax.imshow(cam_resized, cmap='viridis', alpha=0.5, aspect='auto', origin='lower', extent=im.get_extent())
        
        ax.set_title(f"CAM for '{target_label_for_cam}' (Prob: {probabilities[target_idx]:.3f})\nPredicted: '{predicted_label}' ({predicted_prob:.3f}) | Model: {config.model_name} | Sample: {actual_samplename_to_process}")
        ax.set_xlabel("Time Frames (5s chunk)")
        ax.set_ylabel("Mel Frequency Bins")
        plt.tight_layout()

        plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_image_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Error during visualization: {e}")
        sys.exit(1)

# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Class Activation Map (CAM) for BirdCLEF EfficientAT model.")

    parser.add_argument('--checkpoint', type=str, default=None, # Default handled in main function
                        help="Path to the trained model checkpoint (.pth file). If None, tries to use the first .pth in config.MODEL_OUTPUT_DIR.")
    # Removed --spectrogram_npz argument
    parser.add_argument('--samplename', type=str, default=None,
                        help="Specific samplename (key in NPZ) to visualize. If None, a random sample is chosen.")
    # Removed --index argument
    parser.add_argument('--target_class', type=str, default=None,
                        help="Target primary_label to generate CAM for. If None, derived from samplename or uses predicted class.")
    # Removed --output_path argument
    parser.add_argument('--device', type=str, default=config.device, 
                        help="Device to run inference on (e.g., 'cuda', 'cpu').")

    args = parser.parse_args()

    # Checkpoint existence is handled in visualize_cam if default is used.
    # If user provides a checkpoint, check it here.
    if args.checkpoint and not os.path.exists(args.checkpoint):
        print(f"Error: Specified checkpoint file not found at {args.checkpoint}")
        sys.exit(1)
    
    # Spectrogram NPZ path check (using config value)
    if not os.path.exists(config.PREPROCESSED_NPZ_PATH):
        print(f"Error: Spectrogram NPZ file not found at {config.PREPROCESSED_NPZ_PATH} (from config.py)")
        sys.exit(1)

    visualize_cam(args)
    print("--- CAM Visualization Script Finished ---") 