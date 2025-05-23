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
# Update these counts based on the output logs of your preprocess/preprocessing.py runs.
# For example, from "Successfully generated spectrograms for X primary files."
EXPECTED_TRAIN_NPZ_FILE_COUNT = 28568  # Placeholder: e.g., from train mode log
EXPECTED_VAL_NPZ_FILE_COUNT = 25353    # From your provided val mode log

# Add a few samplenames that you know were logged with "VAL_MODE_SKIP"
# These should NOT be present as keys in the val_npz file.
# Example from your log: "1194042-CSA18783: VAL_MODE_SKIP" -> samplename is "1194042-CSA18783"
KNOWN_VAL_MODE_SKIPPED_SAMPLENAMES = [
    "1194042-CSA18783",
    "126247-iNat146584",
    "1346504-CSA18784"
]
# --- End Test Configuration ---

class TestPreprocessingOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = config
        cls.model_name = cls.config.model_name
        cls.preprocessed_dir = cls.config._PREPROCESSED_OUTPUT_DIR

        cls.train_npz_filename = f"spectrograms_{cls.model_name}_train.npz"
        cls.val_npz_filename = f"spectrograms_{cls.model_name}_val.npz"

        cls.train_npz_path = os.path.join(cls.preprocessed_dir, cls.train_npz_filename)
        cls.val_npz_path = os.path.join(cls.preprocessed_dir, cls.val_npz_filename)

        cls.train_data = None
        cls.val_data = None

        if not os.path.exists(cls.train_npz_path):
            raise FileNotFoundError(f"Train NPZ file not found: {cls.train_npz_path}. "
                                    f"Please run 'python preprocess/preprocessing.py --mode train'")
        cls.train_data = np.load(cls.train_npz_path, allow_pickle=True)

        if not os.path.exists(cls.val_npz_path):
            raise FileNotFoundError(f"Validation NPZ file not found: {cls.val_npz_path}. "
                                    f"Please run 'python preprocess/preprocessing.py --mode val'")
        cls.val_data = np.load(cls.val_npz_path, allow_pickle=True)

    def test_01_files_exist_and_load(self):
        self.assertTrue(os.path.exists(self.train_npz_path), f"Train NPZ file {self.train_npz_path} should exist.")
        self.assertIsNotNone(self.train_data, "Train NPZ data should load.")
        self.assertTrue(os.path.exists(self.val_npz_path), f"Validation NPZ file {self.val_npz_path} should exist.")
        self.assertIsNotNone(self.val_data, "Validation NPZ data should load.")

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

    def test_03_val_npz_content_and_shape(self):
        self.assertGreater(len(self.val_data.files), 0,
                         f"Validation NPZ ({self.val_npz_filename}) should not be empty.")

        # Check the number of files/keys
        if EXPECTED_VAL_NPZ_FILE_COUNT > 0:
            self.assertEqual(len(self.val_data.files), EXPECTED_VAL_NPZ_FILE_COUNT,
                             f"Number of entries in {self.val_npz_filename} does not match expected count. "
                             f"Update EXPECTED_VAL_NPZ_FILE_COUNT in test script from logs.")

        # Wrap the iterator with tqdm for a progress bar
        for key in tqdm(self.val_data.files, desc=f"Testing {self.val_npz_filename} contents"):
            spec_array = self.val_data[key]
            self.assertIsInstance(spec_array, np.ndarray, f"Val data for key '{key}' should be a numpy array.")
            self.assertEqual(spec_array.dtype, np.float32, f"Val data for key '{key}' should be float32.")
            self.assertFalse(np.isnan(spec_array).any(), f"Val data for key '{key}' should not contain NaNs.")
            self.assertFalse(np.isinf(spec_array).any(), f"Val data for key '{key}' should not contain Infs.")

            self.assertEqual(len(spec_array.shape), 3, f"Val spec_array for '{key}' should have 3 dims (versions, H, W).")
            self.assertEqual(spec_array.shape[0], 1, f"Val spec_array for '{key}' should have exactly 1 version.")
            self.assertEqual(spec_array.shape[1], self.config.PREPROCESS_TARGET_SHAPE[0],
                             f"Val spec height for '{key}' is {spec_array.shape[1]}, expected {self.config.PREPROCESS_TARGET_SHAPE[0]}.")
            self.assertEqual(spec_array.shape[2], self.config.PREPROCESS_TARGET_SHAPE[1],
                             f"Val spec width for '{key}' is {spec_array.shape[2]}, expected {self.config.PREPROCESS_TARGET_SHAPE[1]}.")

    def test_04_val_npz_skipped_keys_not_present(self):
        if not KNOWN_VAL_MODE_SKIPPED_SAMPLENAMES:
            self.skipTest("No KNOWN_VAL_MODE_SKIPPED_SAMPLENAMES defined for testing.")

        val_keys = set(self.val_data.files)
        for skipped_key in KNOWN_VAL_MODE_SKIPPED_SAMPLENAMES:
            self.assertNotIn(skipped_key, val_keys,
                             f"Key '{skipped_key}' was expected to be skipped in val mode but was found in {self.val_npz_filename}.")

    # Optional: Add a test for value ranges if you have a clear expectation
    # def test_05_spectrogram_value_ranges(self):
    #     # Example: Assuming normalization to roughly [0, 1] by AugmentMelSTFT's (melspec + 4.5) / 5.
    #     # This might need adjustment based on the exact normalization.
    #     for key_type, data_dict in [("Train", self.train_data), ("Val", self.val_data)]:
    #         for key in tqdm(data_dict.files, desc=f"Testing {key_type} spec value ranges"):
    #             spec_array = data_dict[key]
    #             # A broader range might be safer unless normalization is strictly to [0,1]
    #             self.assertTrue(-0.5 <= spec_array.min() <= 0.5,  # Looser check around 0
    #                             f"{key_type} spec min value out of expected range for {key}: {spec_array.min()}")
    #             self.assertTrue(0.5 <= spec_array.max() <= 1.5,  # Looser check around 1
    #                             f"{key_type} spec max value out of expected range for {key}: {spec_array.max()}")
    #             # A more precise check if normalization is strictly [0,1]:
    #             # self.assertGreaterEqual(spec_array.min(), 0.0, f"{key_type} spec min value for {key}: {spec_array.min()}")
    #             # self.assertLessEqual(spec_array.max(), 1.0, f"{key_type} spec max value for {key}: {spec_array.max()}")


if __name__ == '__main__':
    unittest.main()
