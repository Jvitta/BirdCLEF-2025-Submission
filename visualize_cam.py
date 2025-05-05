import os
import argparse
import random
import warnings
import logging
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Assuming 'config.py' and 'birdclef_training.py' are accessible
from config import config
from birdclef_training import BirdCLEFModel # Use the model definition from training

# Suppress warnings and limit logging output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# --- CAM Helper Functions ---

# Global variables to store hook outputs
activations = None
gradients = None # Not needed for basic CAM, but good for Grad-CAM

def forward_hook(module, input, output):
    """Hook to capture the output feature maps of a layer."""
    global activations
    activations = output

# --- Main Visualization Function ---

def visualize_cam(args):
    """Loads model, data, computes CAM, and saves visualization."""
    print(f"--- Starting CAM Visualization ---")
    print(f"Using device: {args.device}")
    print(f"Model Checkpoint: {args.checkpoint}")
    print(f"Spectrogram NPZ: {args.spectrogram_npz}")
    if args.samplename:
        print(f"Target Samplename: {args.samplename}")
    else:
        print(f"Target Index: {args.index}")
    if args.target_class:
        print(f"Target Class: {args.target_class}")
    else:
        print(f"Target Class: Will use predicted class with highest probability.")
    print(f"Output Path: {args.output_path}")

    # --- 1. Load Taxonomy ---
    print("Loading taxonomy...")
    try:
        taxonomy_df = pd.read_csv(config.taxonomy_path)
        species_ids = taxonomy_df['primary_label'].tolist()
        idx_to_label = {idx: label for idx, label in enumerate(species_ids)}
        label_to_idx = {label: idx for idx, label in enumerate(species_ids)}
        num_classes = len(species_ids)
        if num_classes != config.num_classes:
             print(f"Warning: Taxonomy ({num_classes}) vs config ({config.num_classes}). Using taxonomy.")
             config.num_classes = num_classes # Adjust config in memory if needed
    except Exception as e:
        print(f"Error loading taxonomy: {e}")
        sys.exit(1)

    # --- 2. Load Model ---
    print("Loading model...")
    try:
        # Use the model definition consistent with training
        model = BirdCLEFModel(config) # Pass config to model
        checkpoint = torch.load(args.checkpoint, map_location=torch.device(args.device))
        # Handle potential keys ('model_state_dict' vs direct state_dict)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
             model.load_state_dict(checkpoint) # Assume checkpoint is the state_dict
        model = model.to(args.device)
        model.eval()
        print(f"Model '{config.model_name}' loaded successfully.")
    except Exception as e:
        print(f"Error loading model checkpoint {args.checkpoint}: {e}")
        sys.exit(1)

    # --- 3. Identify Target Layer and Register Hook ---
    # Target the layer *before* the final pooling and classifier.
    # For EfficientNet, this is often the output of the main convolutional block sequence.
    # We can access this via `model.backbone.forward_features` in timm models.
    # Let's hook the final layer *within* the backbone's main feature extractor.
    # Common final conv layer names: `conv_head` (EfficientNet), layer4 (ResNet)
    target_layer = None
    if hasattr(model.backbone, 'conv_head'):
        target_layer = model.backbone.conv_head
        print("Hooking into 'model.backbone.conv_head'")
    elif hasattr(model.backbone, 'layer4'): # Example for ResNets
        target_layer = model.backbone.layer4
        print("Hooking into 'model.backbone.layer4'")
    else:
        # Fallback: Try to inspect common Timm patterns or require manual specification
        print("Warning: Could not automatically determine target layer. Attempting to hook final block...")
        try:
            # This is a guess, might need adjustment based on specific model architecture
            final_block_name = list(model.backbone.blocks.named_children())[-1][0]
            target_layer = getattr(model.backbone.blocks, final_block_name)
            print(f"Hooking into presumed final block: '{final_block_name}'")
        except AttributeError:
             print("ERROR: Failed to find a suitable target layer for CAM hooks. Exiting.")
             print("You may need to manually inspect the model structure and set the target_layer.")
             sys.exit(1)

    if target_layer:
        hook_handle = target_layer.register_forward_hook(forward_hook)
    else:
        # Should have exited above, but safeguard
        print("ERROR: target_layer is None. Cannot register hook.")
        sys.exit(1)

    # --- 4. Load and Preprocess Spectrogram ---
    print("Loading and preprocessing spectrogram...")
    try:
        with np.load(args.spectrogram_npz) as data_archive:
            if args.samplename:
                if args.samplename not in data_archive:
                    print(f"Error: Samplename '{args.samplename}' not found in {args.spectrogram_npz}")
                    available_keys = list(data_archive.keys())[:10]
                    print(f"Available keys (first 10): {available_keys}")
                    sys.exit(1)
                spec_data = data_archive[args.samplename]
            else:
                all_keys = list(data_archive.keys())
                if args.index >= len(all_keys):
                    print(f"Error: Index {args.index} out of bounds for {len(all_keys)} samples in {args.spectrogram_npz}")
                    sys.exit(1)
                selected_key = all_keys[args.index]
                spec_data = data_archive[selected_key]
                print(f"(Selected key at index {args.index}: {selected_key})")


        # Handle multi-chunk vs single-chunk (select first chunk if multiple)
        if spec_data.ndim == 3:
            print(f"Input data has shape {spec_data.shape} (multiple chunks). Selecting first chunk.")
            spec_np = spec_data[0]
        elif spec_data.ndim == 2:
            spec_np = spec_data
        else:
            print(f"Error: Unexpected spectrogram shape {spec_data.shape}. Expected 2 or 3 dimensions.")
            sys.exit(1)
            
        # Ensure correct shape
        if spec_np.shape != tuple(config.TARGET_SHAPE):
             print(f"Warning: Spectrogram shape {spec_np.shape} doesn't match config TARGET_SHAPE {config.TARGET_SHAPE}. Resizing...")
             spec_np = cv2.resize(spec_np, config.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)


        # --- Preprocessing Steps (similar to inference/training) ---
        # 1. Ensure float32
        spec_np_float = spec_np.astype(np.float32)

        # 2. Store original for visualization before normalization
        original_spec_display = spec_np_float.copy() 

        # 3. Convert to Tensor, add channel, repeat channels
        spec_tensor = torch.tensor(spec_np_float).unsqueeze(0).repeat(3, 1, 1)

        # 4. Normalize (using standard ImageNet stats like in training/inference)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        spec_tensor_normalized = normalize(spec_tensor)

        # 5. Add batch dimension and move to device
        input_tensor = spec_tensor_normalized.unsqueeze(0).to(args.device) # Shape: (1, 3, H, W)

        print(f"Spectrogram preprocessed. Tensor shape: {input_tensor.shape}")

    except Exception as e:
        print(f"Error loading or preprocessing spectrogram: {e}")
        sys.exit(1)

    # --- 5. Forward Pass and Get Predictions ---
    print("Performing forward pass...")
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    predicted_idx = np.argmax(probabilities)
    predicted_label = idx_to_label.get(predicted_idx, "Unknown")
    predicted_prob = probabilities[predicted_idx]
    print(f"Prediction: '{predicted_label}' (Index: {predicted_idx}) with probability {predicted_prob:.4f}")

    # Determine target class index for CAM
    if args.target_class:
        if args.target_class in label_to_idx:
            target_idx = label_to_idx[args.target_class]
            target_label = args.target_class
            print(f"Using specified target class: '{target_label}' (Index: {target_idx})")
        else:
            print(f"Error: Specified target class '{args.target_class}' not found in taxonomy.")
            sys.exit(1)
    else:
        target_idx = predicted_idx
        target_label = predicted_label
        print(f"Using predicted class as target: '{target_label}' (Index: {target_idx})")


    # --- 6. Calculate CAM ---
    print("Calculating CAM...")
    if activations is None:
        print("Error: Hook did not capture activations. Cannot compute CAM.")
        hook_handle.remove()
        sys.exit(1)

    try:
        # Get weights of the target class from the final classifier layer
        # Shape: (num_classes, feature_dim) -> We need weights for target_idx -> Shape: (feature_dim,)
        classifier_weights = model.classifier.weight.data # Shape: (num_classes, C)
        target_weights = classifier_weights[target_idx, :] # Shape: (C,)

        # Feature maps from the hooked layer
        # Shape: (batch, channels, height, width) -> (1, C, H, W) for our case
        feature_maps = activations.squeeze(0) # Shape: (C, H, W)

        # Calculate weighted sum of feature maps: C * (C, H, W) -> (H, W)
        # Option 1: Einsum (often cleaner)
        # cam = torch.einsum('c,chw->hw', target_weights, feature_maps)

        # Option 2: Reshape and MatMul (more explicit)
        C, H, W = feature_maps.shape
        feature_maps_reshaped = feature_maps.view(C, H * W) # Shape: (C, H*W)
        cam = torch.matmul(target_weights.unsqueeze(0), feature_maps_reshaped) # (1, C) @ (C, H*W) -> (1, H*W)
        cam = cam.view(H, W) # Shape: (H, W)

        # Apply ReLU (common practice for CAM)
        cam = F.relu(cam)

        # Normalize CAM to 0-1 range for visualization
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-6) # Add epsilon for numerical stability

        cam = cam.cpu().numpy() # Convert to NumPy array

        # Resize CAM to original spectrogram dimensions
        original_height, original_width = original_spec_display.shape
        cam_resized = cv2.resize(cam, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

        print(f"CAM calculated and resized to {cam_resized.shape}")

    except Exception as e:
        print(f"Error during CAM calculation: {e}")
        hook_handle.remove() # Clean up hook
        sys.exit(1)
    finally:
         hook_handle.remove() # Always remove hook afterwards

    # --- 7. Visualize and Save ---
    print("Generating visualization...")
    try:
        plt.style.use('default') # Reset to default style
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Display original spectrogram
        # Use origin='lower' if frequency axis is bottom-up, 'upper' if top-down
        # Check your preprocessing steps to confirm orientation
        im = ax.imshow(original_spec_display, aspect='auto', origin='lower', cmap='magma') 
        fig.colorbar(im, ax=ax, format='%+2.0f dB' if 'db' in args.spectrogram_npz.lower() else None, label='Amplitude/Power') # Optional dB scaling hint

        # Overlay CAM heatmap
        # Use alpha for transparency, choose a perceptually uniform colormap like 'viridis' or 'plasma'
        heatmap = ax.imshow(cam_resized, cmap='viridis', alpha=0.5, aspect='auto', origin='lower', extent=im.get_extent())
        # fig.colorbar(heatmap, ax=ax, label='Activation Intensity') # Can add separate colorbar for CAM if needed

        ax.set_title(f"CAM for '{target_label}' (Prob: {probabilities[target_idx]:.3f})
Predicted: '{predicted_label}' ({predicted_prob:.3f}) | File: {args.checkpoint.split('/')[-1]} | Sample: {args.samplename or selected_key}")
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Mel Frequency Bins")
        plt.tight_layout()

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        plt.savefig(args.output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {args.output_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Error during visualization: {e}")
        sys.exit(1)

# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Class Activation Map (CAM) for a BirdCLEF model.")

    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument('--spectrogram_npz', type=str, required=True,
                        help="Path to the preprocessed spectrogram NPZ file.")
    parser.add_argument('--samplename', type=str, default=None,
                        help="Specific samplename (key in NPZ) to visualize. Use instead of --index.")
    parser.add_argument('--index', type=int, default=0,
                        help="Index of the spectrogram in the NPZ file to visualize (if --samplename is not provided).")
    parser.add_argument('--target_class', type=str, default=None,
                        help="Target primary_label to generate CAM for. If None, uses the predicted class.")
    parser.add_argument('--output_path', type=str, default="cam_visualization.png",
                        help="Path to save the output visualization PNG file.")
    parser.add_argument('--device', type=str, default=config.device, # Use device from config by default
                        help="Device to run inference on (e.g., 'cuda', 'cpu').")

    args = parser.parse_args()

    # Basic validation
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.spectrogram_npz):
        print(f"Error: Spectrogram NPZ file not found at {args.spectrogram_npz}")
        sys.exit(1)
    if args.samplename is None and args.index < 0:
         print(f"Error: Index must be non-negative if samplename is not provided.")
         sys.exit(1)

    visualize_cam(args)
    print("--- CAM Visualization Script Finished ---") 