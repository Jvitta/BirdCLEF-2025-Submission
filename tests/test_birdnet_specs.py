import unittest
import os
import sys
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

# Add project root to sys.path to allow importing 'config'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import config

# --- Test Configuration ---
# !! IMPORTANT !!
# Update these counts based on the output logs of your preprocess/pseudo/birdnet_specs.py runs.
# This should match the "Saving X pseudo label spectrograms" message, not necessarily "Successfully generated Y".
# The difference can be due to duplicate segment_keys if multiple pseudo-labels map to the same key.
EXPECTED_PSEUDO_TRAIN_NPZ_FILE_COUNT = 25512  # From your provided train mode log ("Saving ...")
EXPECTED_PSEUDO_VAL_NPZ_FILE_COUNT = 25512    # From your provided val mode log ("Saving ...")
SAMPLE_VISUALIZATIONS_COUNT = 3  # Number of samples to visualize from each NPZ
# --- End Test Configuration ---

class TestBirdnetSpecsOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = config
        cls.model_name = cls.config.model_name
        cls.preprocessed_dir = cls.config._PREPROCESSED_OUTPUT_DIR

        cls.train_npz_filename = f"pseudo_spectrograms_{cls.model_name}_train.npz"
        cls.val_npz_filename = f"pseudo_spectrograms_{cls.model_name}_val.npz"

        cls.train_npz_path = os.path.join(cls.preprocessed_dir, cls.train_npz_filename)
        cls.val_npz_path = os.path.join(cls.preprocessed_dir, cls.val_npz_filename)

        cls.train_data = None
        cls.val_data = None

        if not os.path.exists(cls.train_npz_path):
            raise FileNotFoundError(f"Pseudo Train NPZ file not found: {cls.train_npz_path}. "
                                    f"Please run 'python preprocess/pseudo/birdnet_specs.py --mode train'")
        cls.train_data = np.load(cls.train_npz_path, allow_pickle=True)

        if not os.path.exists(cls.val_npz_path):
            raise FileNotFoundError(f"Pseudo Validation NPZ file not found: {cls.val_npz_path}. "
                                    f"Please run 'python preprocess/pseudo/birdnet_specs.py --mode val'")
        cls.val_data = np.load(cls.val_npz_path, allow_pickle=True)

    def test_01_files_exist_and_load(self):
        self.assertTrue(os.path.exists(self.train_npz_path), f"Pseudo Train NPZ file {self.train_npz_path} should exist.")
        self.assertIsNotNone(self.train_data, "Pseudo Train NPZ data should load.")
        self.assertTrue(os.path.exists(self.val_npz_path), f"Pseudo Val NPZ file {self.val_npz_path} should exist.")
        self.assertIsNotNone(self.val_data, "Pseudo Val NPZ data should load.")

    def _assert_npz_content_and_shape(self, data_dict, filename, expected_count, mode_label):
        self.assertGreater(len(data_dict.files), 0, 
                         f"Pseudo {mode_label} NPZ ({filename}) should not be empty.")
        
        if expected_count > 0:
             self.assertEqual(len(data_dict.files), expected_count,
                             f"Number of entries in {filename} ({len(data_dict.files)}) does not match expected count ({expected_count}). "
                             f"Update EXPECTED_PSEUDO_{mode_label.upper()}_NPZ_FILE_COUNT in test script from logs (use 'Saving X ...' count).")

        for key in tqdm(data_dict.files, desc=f"Testing {filename} contents"):
            spec_array = data_dict[key]
            self.assertIsInstance(spec_array, np.ndarray, f"Pseudo {mode_label} data for key '{key}' should be a numpy array.")
            self.assertEqual(spec_array.dtype, np.float32, f"Pseudo {mode_label} data for key '{key}' should be float32.")
            self.assertFalse(np.isnan(spec_array).any(), f"Pseudo {mode_label} data for key '{key}' should not contain NaNs.")
            self.assertFalse(np.isinf(spec_array).any(), f"Pseudo {mode_label} data for key '{key}' should not contain Infs.")

            self.assertEqual(len(spec_array.shape), 3, f"Pseudo {mode_label} spec_array for '{key}' should have 3 dims (versions, H, W).")
            self.assertEqual(spec_array.shape[0], 1, f"Pseudo {mode_label} spec_array for '{key}' should have exactly 1 version.")
            self.assertEqual(spec_array.shape[1], self.config.PREPROCESS_TARGET_SHAPE[0],
                             f"Pseudo {mode_label} spec height for '{key}' is {spec_array.shape[1]}, expected {self.config.PREPROCESS_TARGET_SHAPE[0]}.")
            self.assertEqual(spec_array.shape[2], self.config.PREPROCESS_TARGET_SHAPE[1],
                             f"Pseudo {mode_label} spec width for '{key}' is {spec_array.shape[2]}, expected {self.config.PREPROCESS_TARGET_SHAPE[1]}.")

            # Check the actual 2D spectrogram content (spec_array[0] as shape[0] is 1)
            actual_spec_2d = spec_array[0]
            mean_val = actual_spec_2d.mean()
            std_val = actual_spec_2d.std()

            # Check if the spectrogram is not a constant array (e.g., all zeros)
            self.assertGreater(std_val, 1e-6, 
                               f"Pseudo {mode_label} spec_array for '{key}' has very low standard deviation ({std_val:.2e}), "
                               f"suggesting it might be a constant array. Mean: {mean_val:.2e}.")
            
            # If std is extremely low, and mean is also very low, it's problematic.
            # This is somewhat covered by the above, but an explicit check for near-zero mean if std is low can be useful.
            if std_val < 1e-5: # If it's almost constant
                self.assertNotAlmostEqual(mean_val, 0.0, places=5, 
                                          msg=f"Pseudo {mode_label} spec_array for '{key}' is nearly constant and mean is close to zero. "
                                              f"Mean: {mean_val:.2e}, Std: {std_val:.2e}.")

    def test_02_pseudo_train_npz_content_and_shape(self):
        self._assert_npz_content_and_shape(self.train_data, self.train_npz_filename, 
                                           EXPECTED_PSEUDO_TRAIN_NPZ_FILE_COUNT, "train")

    def test_03_pseudo_val_npz_content_and_shape(self):
        self._assert_npz_content_and_shape(self.val_data, self.val_npz_filename, 
                                         EXPECTED_PSEUDO_VAL_NPZ_FILE_COUNT, "val")

    def test_04_visualize_sample_spectrograms(self):
        output_visualization_dir = os.path.join(self.config.OUTPUT_DIR, "test_outputs", "birdnet_specs_visualizations")
        os.makedirs(output_visualization_dir, exist_ok=True)
        print(f"Saving spectrogram visualizations to: {output_visualization_dir}")

        for data_dict, npz_filename_short, mode_label in [
            (self.train_data, self.train_npz_filename, "train"),
            (self.val_data, self.val_npz_filename, "val")
        ]:
            if not data_dict or not data_dict.files:
                print(f"Skipping visualization for {mode_label} as data is not loaded or empty.")
                continue

            all_keys = list(data_dict.keys())
            if not all_keys:
                print(f"No keys found in {npz_filename_short} for visualization.")
                continue
            
            selected_keys = random.sample(all_keys, min(len(all_keys), SAMPLE_VISUALIZATIONS_COUNT))

            for i, key in enumerate(tqdm(selected_keys, desc=f"Visualizing {mode_label} samples from {npz_filename_short}")):
                spec_array_3d = data_dict[key]
                # As established, pseudo specs are (1, H, W)
                if spec_array_3d.ndim == 3 and spec_array_3d.shape[0] == 1:
                    spec_array_2d = spec_array_3d[0]
                else:
                    print(f"Skipping visualization for key '{key}' in {npz_filename_short} due to unexpected shape: {spec_array_3d.shape}")
                    continue

                plt.figure(figsize=(10, 4))
                plt.imshow(spec_array_2d, aspect='auto', origin='lower', cmap='magma')
                plt.colorbar(label='Amplitude')
                plt.title(f"{mode_label.capitalize()} Spectrogram: {key}")
                plt.xlabel("Time Frames")
                plt.ylabel("Mel Bins")
                
                # Sanitize key for filename
                safe_key_filename = "".join(c if c.isalnum() else "_" for c in key)
                if len(safe_key_filename) > 50: # Truncate if too long
                    safe_key_filename = safe_key_filename[:50]

                save_path = os.path.join(output_visualization_dir, f"{mode_label}_{safe_key_filename}_{i}.png")
                try:
                    plt.savefig(save_path)
                except Exception as e:
                    print(f"Error saving plot for {key} to {save_path}: {e}")
                plt.close()
            print(f"Finished visualizing {len(selected_keys)} samples for {mode_label}.")

    # Optional: Add a test for value ranges, similar to test_preprocessing.py
    # def test_04_spectrogram_value_ranges(self):
    #     for key_type, data_dict in [("Pseudo Train", self.train_data), ("Pseudo Val", self.val_data)]:
    #         for key in tqdm(data_dict.files, desc=f"Testing {key_type} spec value ranges"):
    #             spec_array = data_dict[key]
    #             # Check based on your expected normalization in birdnet_specs.py
    #             # If using AugmentMelSTFT: (melspec + 4.5) / 5. 
    #             self.assertTrue(-0.5 <= spec_array.min() <= 0.5, 
    #                             f"{key_type} spec min value out of expected range for {key}: {spec_array.min()}")
    #             self.assertTrue(0.5 <= spec_array.max() <= 1.5, 
    #                             f"{key_type} spec max value out of expected range for {key}: {spec_array.max()}")

if __name__ == '__main__':
    unittest.main()
