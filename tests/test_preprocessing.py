import unittest
import os
import sys
import numpy as np
from tqdm import tqdm

# Add project root to sys.path to allow importing 'config'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import config

# --- Test Configuration ---
# Update these counts based on the output logs of your preprocess/preprocessing.py --mode train runs.
# For example, from "Successfully generated spectrograms for X primary files."
EXPECTED_TRAIN_NPZ_FILE_COUNT = 28566  # Placeholder: e.g., from train mode log
# EXPECTED_VAL_NPZ_FILE_COUNT = 25353    # Removed, as val NPZ from main preprocessing is not used

# Add a few samplenames that you know were logged with "VAL_MODE_SKIP"
# These should NOT be present as keys in the val_npz file.
# This test is being removed as we are no longer testing the _val.npz from main preprocessing.
# KNOWN_VAL_MODE_SKIPPED_SAMPLENAMES = [
#     "1194042-CSA18783",
#     "126247-iNat146584",
#     "1346504-CSA18784"
# ]
# --- End Test Configuration ---

class TestPreprocessingOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = config
        cls.model_name = cls.config.model_name
        cls.preprocessed_dir = cls.config._PREPROCESSED_OUTPUT_DIR

        cls.train_npz_filename = f"spectrograms_{cls.model_name}_train.npz"
        # cls.val_npz_filename = f"spectrograms_{cls.model_name}_val.npz" # Removed

        cls.train_npz_path = os.path.join(cls.preprocessed_dir, cls.train_npz_filename)
        # cls.val_npz_path = os.path.join(cls.preprocessed_dir, cls.val_npz_filename) # Removed

        cls.train_data = None
        # cls.val_data = None # Removed

        if not os.path.exists(cls.train_npz_path):
            raise FileNotFoundError(f"Train NPZ file not found: {cls.train_npz_path}. "
                                    f"Please run 'python preprocess/preprocessing.py --mode train'")
        cls.train_data = np.load(cls.train_npz_path, allow_pickle=True)

        # Removed val_data loading
        # if not os.path.exists(cls.val_npz_path):
        #     raise FileNotFoundError(f"Validation NPZ file not found: {cls.val_npz_path}. "
        #                             f"Please run 'python preprocess/preprocessing.py --mode val'")
        # cls.val_data = np.load(cls.val_npz_path, allow_pickle=True)

    def test_01_files_exist_and_load(self):
        self.assertTrue(os.path.exists(self.train_npz_path), f"Train NPZ file {self.train_npz_path} should exist.")
        self.assertIsNotNone(self.train_data, "Train NPZ data should load.")
        # self.assertTrue(os.path.exists(self.val_npz_path), f"Validation NPZ file {self.val_npz_path} should exist.") # Removed
        # self.assertIsNotNone(self.val_data, "Validation NPZ data should load.") # Removed

    def test_02_train_npz_content_and_shape(self):
        self.assertGreater(len(self.train_data.files), 0, 
                         f"Train NPZ ({self.train_npz_filename}) should not be empty.")
        
        # Check the number of files/keys if a placeholder is set
        if EXPECTED_TRAIN_NPZ_FILE_COUNT > 0:
             self.assertEqual(len(self.train_data.files), EXPECTED_TRAIN_NPZ_FILE_COUNT,
                             f"Number of entries in {self.train_npz_filename} does not match expected count. "
                             f"Update EXPECTED_TRAIN_NPZ_FILE_COUNT in test script from logs.")

        # Wrap the iterator with tqdm for a progress bar
        for key in tqdm(self.train_data.files, desc=f"Testing {self.train_npz_filename} contents"):
            spec_array = self.train_data[key]
            self.assertIsInstance(spec_array, np.ndarray, f"Train data for key '{key}' should be a numpy array.")
            self.assertEqual(spec_array.dtype, np.float32, f"Train data for key '{key}' should be float32.")
            self.assertFalse(np.isnan(spec_array).any(), f"Train data for key '{key}' should not contain NaNs.")
            self.assertFalse(np.isinf(spec_array).any(), f"Train data for key '{key}' should not contain Infs.")

            self.assertEqual(len(spec_array.shape), 3, f"Train spec_array for '{key}' should have 3 dims (versions, H, W).")
            self.assertGreaterEqual(spec_array.shape[0], 1, f"Train spec_array for '{key}' should have at least 1 version.")
            # Further checks on num_versions could be:
            # if self.config.DYNAMIC_CHUNK_COUNTING:
            #     self.assertLessEqual(spec_array.shape[0], self.config.MAX_CHUNKS_RARE)
            #     self.assertGreaterEqual(spec_array.shape[0], self.config.MIN_CHUNKS_COMMON) # This might be too strict
            # else:
            #     self.assertEqual(spec_array.shape[0], self.config.PRECOMPUTE_VERSIONS) # If relevant

            self.assertEqual(spec_array.shape[1], self.config.PREPROCESS_TARGET_SHAPE[0],
                             f"Train spec height for '{key}' is {spec_array.shape[1]}, expected {self.config.PREPROCESS_TARGET_SHAPE[0]}.")
            self.assertEqual(spec_array.shape[2], self.config.PREPROCESS_TARGET_SHAPE[1],
                             f"Train spec width for '{key}' is {spec_array.shape[2]}, expected {self.config.PREPROCESS_TARGET_SHAPE[1]}.")

    # test_03_val_npz_content_and_shape and test_04_val_npz_skipped_keys_not_present are removed.

    # Optional: Add a test for value ranges if you have a clear expectation
    # def test_03_spectrogram_value_ranges(self): # Renumber if keeping
    #     # Example: Assuming normalization to roughly [0, 1] by AugmentMelSTFT's (melspec + 4.5) / 5.
    #     # This might need adjustment based on the exact normalization.
    #     # Only testing train_data now
    #     for key in tqdm(self.train_data.files, desc=f"Testing Train spec value ranges"):
    #         spec_array = self.train_data[key]
    #         # A broader range might be safer unless normalization is strictly to [0,1]
    #         self.assertTrue(-0.5 <= spec_array.min() <= 0.5,  # Looser check around 0
    #                         f"Train spec min value out of expected range for {key}: {spec_array.min()}")
    #         self.assertTrue(0.5 <= spec_array.max() <= 1.5,  # Looser check around 1
    #                         f"Train spec max value out of expected range for {key}: {spec_array.max()}")
    #         # A more precise check if normalization is strictly [0,1]:
    #         # self.assertGreaterEqual(spec_array.min(), 0.0, f"Train spec min value for {key}: {spec_array.min()}")
    #         # self.assertLessEqual(spec_array.max(), 1.0, f"Train spec max value for {key}: {spec_array.max()}")

if __name__ == '__main__':
    unittest.main()
