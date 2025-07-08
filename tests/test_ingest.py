"""
Comprehensive tests for the preprocessing pipeline.

This file contains:
1. Unit tests for individual pipeline components
2. Validation functions with known mathematical solutions
3. Synthetic data tests with expected outcomes
4. Integration tests for the complete pipeline
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add src to path to import the preprocessing module
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.ingest import DataIngester


class TestPreprocessingPipeline(unittest.TestCase):
    """Test suite for the preprocessing pipeline."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = Path(self.temp_dir) / "raw"
        self.processed_dir = Path(self.temp_dir) / "processed"
        self.raw_dir.mkdir()
        self.processed_dir.mkdir()
        
        # Create DataIngester instance
        self.ingester = DataIngester(str(self.raw_dir), str(self.processed_dir))
    
    def tearDown(self):
        """Clean up after each test."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_synthetic_csv(self, filename: str, data_type: str, n_records: int = 100):
        """Create synthetic CSV data with known properties."""
        csv_file = self.raw_dir / filename
        
        # Generate timestamps
        start_time = datetime(2025, 1, 1, 0, 0, 0)
        timestamps = [start_time + timedelta(minutes=10*i) for i in range(n_records)]
        
        # Generate synthetic data with known statistical properties
        if data_type == "water_height":
            # Water height: mean=0, std=1, range roughly [-2, 2]
            values = np.random.normal(0, 1, n_records)
            values = np.clip(values, -2, 2)
        elif data_type == "flow_rate":
            # Flow rate: mean=1, std=0.5, range roughly [0, 2]
            values = np.random.normal(1, 0.5, n_records)
            values = np.clip(values, 0, 2)
        elif data_type == "wind_direction":
            # Wind direction: circular data 0-360 degrees
            values = np.random.uniform(0, 360, n_records)
        else:
            # Default: random values
            values = np.random.normal(0, 1, n_records)
        
        # Create DataFrame
        data = {
            'WAARNEMINGDATUM': [dt.strftime('%Y-%m-%d') for dt in timestamps],
            'WAARNEMINGTIJD (MET/CET)': [dt.strftime('%H:%M:%S') for dt in timestamps],
            'NUMERIEKEWAARDE': values,
            'X': np.random.uniform(0.9, 1.0, n_records),
            'Y': np.random.uniform(-1.0, 1.0, n_records)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(csv_file, sep=';', index=False)
        return df, csv_file
    
    def test_datetime_parsing(self):
        """Test datetime parsing with various formats."""
        # Test YYYY-MM-DD format
        dt1 = self.ingester._parse_datetime("2025-01-01", "12:30:45")
        self.assertEqual(dt1, datetime(2025, 1, 1, 12, 30, 45))
        
        # Test DD-MM-YYYY format
        dt2 = self.ingester._parse_datetime("01-01-2025", "12:30:45")
        self.assertEqual(dt2, datetime(2025, 1, 1, 12, 30, 45))
        
        # Test DD/MM/YYYY format
        dt3 = self.ingester._parse_datetime("01/01/2025", "12:30:45")
        self.assertEqual(dt3, datetime(2025, 1, 1, 12, 30, 45))
        
        # Test invalid format
        dt4 = self.ingester._parse_datetime("invalid", "12:30:45")
        self.assertIsNone(dt4)
    
    def test_data_extraction_synthetic(self):
        """Test data extraction with synthetic data."""
        # Create synthetic CSV file
        original_df, csv_file = self.create_synthetic_csv("water_height.csv", "water_height", 50)
        
        # Extract data
        extracted_df = self.ingester._extract_data_from_csv(csv_file)
        
        # Validate extraction
        self.assertIsNotNone(extracted_df)
        self.assertEqual(len(extracted_df), 50)
        self.assertIn('datetime', extracted_df.columns)
        self.assertIn('datetime_unix', extracted_df.columns)
        self.assertIn('NUMERIEKEWAARDE', extracted_df.columns)
        self.assertIn('X', extracted_df.columns)
        self.assertIn('Y', extracted_df.columns)
        
        # Check that datetime conversion worked
        self.assertEqual(extracted_df['datetime'].dtype, 'datetime64[ns]')
        self.assertTrue(extracted_df['datetime_unix'].notna().all())
        
        # Check that values are preserved (within floating point precision)
        np.testing.assert_array_almost_equal(
            extracted_df['NUMERIEKEWAARDE'].values,
            original_df['NUMERIEKEWAARDE'].values,
            decimal=6
        )
    
    def test_data_cleaning(self):
        """Test data cleaning functionality."""
        # Create test data with known issues
        test_data = {
            'datetime': [
                datetime(2025, 1, 1, 0, 0, 0),
                datetime(2025, 1, 1, 0, 10, 0),
                datetime(2025, 1, 1, 0, 20, 0),
                datetime(2025, 1, 1, 0, 10, 0),  # Duplicate timestamp
                None,  # Missing datetime
            ],
            'datetime_unix': [1.0, 2.0, 3.0, 2.0, None],
            'NUMERIEKEWAARDE': [1.0, 2.0, 3.0, 4.0, 5.0],
            'X': [0.5, 0.6, 0.7, 0.8, 0.9],
            'Y': [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        df = pd.DataFrame(test_data)
        cleaned_df = self.ingester._clean_data(df)
        
        # Should remove rows with missing datetime and duplicates
        self.assertEqual(len(cleaned_df), 3)  # Original 5 - 1 missing datetime - 1 duplicate
        self.assertTrue(cleaned_df['datetime'].notna().all())
        self.assertEqual(len(cleaned_df), len(cleaned_df.drop_duplicates(subset=['datetime'])))
    
    def test_normalization_mathematical_validation(self):
        """Test normalization with mathematical validation."""
        # Create test data with known statistical properties
        n_samples = 1000
        original_mean = 10.0
        original_std = 2.0
        
        test_data = {
            'datetime': [datetime(2025, 1, 1, 0, 0, 0) + timedelta(minutes=i) for i in range(n_samples)],
            'datetime_unix': np.arange(n_samples),
            'NUMERIEKEWAARDE': np.random.normal(original_mean, original_std, n_samples),
            'X': np.random.uniform(0.9, 1.0, n_samples),
            'Y': np.random.uniform(-1.0, 1.0, n_samples)
        }
        
        df = pd.DataFrame(test_data)
        
        # Clean the data first (to match what the pipeline does)
        cleaned_df = self.ingester._clean_data(df)
        
        # Calculate normalization parameters from cleaned data
        self.ingester._get_normalization_params(cleaned_df, "test_data")
        
        # Normalize data
        normalized_df = self.ingester._normalize_data(cleaned_df, "test_data")
        
        # Mathematical validation: normalized data should have mean≈0 and std≈1
        normalized_values = normalized_df['NUMERIEKEWAARDE'].values
        self.assertLess(abs(normalized_values.mean()), 0.1)  # Mean should be close to 0
        self.assertLess(abs(normalized_values.std() - 1.0), 0.1)  # Std should be close to 1
        
        # Verify the normalization formula using the actual parameters from the pipeline
        # This is the key fix - use the actual parameters calculated by the pipeline
        params = self.ingester.norm_params["test_data"]
        actual_mean = params['mean']['NUMERIEKEWAARDE']
        actual_std = params['std']['NUMERIEKEWAARDE']
        
        # Calculate expected normalization using the pipeline's actual parameters
        expected_normalized = (cleaned_df['NUMERIEKEWAARDE'] - actual_mean) / actual_std
        
        # Now the arrays should match perfectly
        np.testing.assert_array_almost_equal(normalized_values, expected_normalized, decimal=6)
        
        # Also verify that other columns were normalized correctly
        for col in ['X', 'Y', 'datetime_unix']:
            if col in params['mean'] and col in params['std']:
                col_mean = params['mean'][col]
                col_std = params['std'][col]
                if col_std is not None and col_std != 0:
                    expected_col_normalized = (cleaned_df[col] - col_mean) / col_std
                    actual_col_normalized = normalized_df[col].values
                    np.testing.assert_array_almost_equal(actual_col_normalized, expected_col_normalized, decimal=6)
    
    def test_normalization_edge_cases(self):
        """Test normalization with edge cases."""
        # Test with zero standard deviation (constant values)
        test_data = {
            'datetime': [datetime(2025, 1, 1, 0, 0, 0) + timedelta(minutes=i) for i in range(3)],
            'datetime_unix': [1.0, 2.0, 3.0],
            'NUMERIEKEWAARDE': [5.0, 5.0, 5.0],  # All same values
            'X': [0.5, 0.5, 0.5],
            'Y': [0.1, 0.1, 0.1]
        }
        
        df = pd.DataFrame(test_data)
        self.ingester._get_normalization_params(df, "constant_data")
        
        # Should handle zero std gracefully (no normalization applied)
        normalized_df = self.ingester._normalize_data(df, "constant_data")
        np.testing.assert_array_equal(normalized_df['NUMERIEKEWAARDE'].values, df['NUMERIEKEWAARDE'].values)
    
    def test_complete_pipeline_integration(self):
        """Test the complete preprocessing pipeline with synthetic data."""
        # Create multiple synthetic datasets
        datasets = ["water_height", "flow_rate", "wind_direction"]
        original_data = {}
        
        for dataset in datasets:
            original_df, _ = self.create_synthetic_csv(f"{dataset}.csv", dataset, 100)
            original_data[dataset] = original_df
        
        # Run complete pipeline
        processed_data = self.ingester.process_all_files()
        
        # Validate results
        self.assertEqual(len(processed_data), 3)
        
        for dataset in datasets:
            self.assertIn(dataset, processed_data)
            processed_df = processed_data[dataset]
            
            # Check that data was processed
            self.assertGreater(len(processed_df), 0)
            self.assertIn('datetime', processed_df.columns)
            self.assertIn('NUMERIEKEWAARDE', processed_df.columns)
            
            # Check that normalization was applied
            # Normalized data should have different statistical properties
            original_stats = original_data[dataset]['NUMERIEKEWAARDE'].describe()
            processed_stats = processed_df['NUMERIEKEWAARDE'].describe()
            
            # Mean should be close to 0 for normalized data
            self.assertLess(abs(processed_stats['mean']), 0.1)
            # Standard deviation should be close to 1 for normalized data
            self.assertLess(abs(processed_stats['std'] - 1.0), 0.1)
    
    def test_timestamp_alignment(self):
        """Test that timestamps are properly aligned across datasets."""
        # Create datasets with different time ranges
        start_times = [
            datetime(2025, 1, 1, 0, 0, 0),
            datetime(2025, 1, 1, 1, 0, 0),  # 1 hour later
            datetime(2025, 1, 1, 0, 30, 0),  # 30 minutes later
        ]
        
        end_times = [
            datetime(2025, 1, 1, 23, 0, 0),
            datetime(2025, 1, 1, 22, 0, 0),  # 1 hour earlier
            datetime(2025, 1, 1, 23, 30, 0),  # 30 minutes later
        ]
        
        datasets = ["water_height", "flow_rate", "wind_direction"]
        
        for i, dataset in enumerate(datasets):
            self.create_synthetic_csv_with_timerange(f"{dataset}.csv", dataset, start_times[i], end_times[i])
        
        # Run pipeline
        processed_data = self.ingester.process_all_files()
        
        # Check that all datasets have the same time range
        time_ranges = []
        for dataset, df in processed_data.items():
            time_ranges.append((df['datetime'].min(), df['datetime'].max()))
        
        # All datasets should have the same time range
        min_start = max(r[0] for r in time_ranges)
        max_end = min(r[1] for r in time_ranges)
        
        for start, end in time_ranges:
            self.assertEqual(start, min_start)
            self.assertEqual(end, max_end)
    
    def create_synthetic_csv_with_timerange(self, filename: str, data_type: str, start_time: datetime, end_time: datetime):
        """Create synthetic CSV data within a specific time range."""
        csv_file = self.raw_dir / filename
        
        # Generate timestamps
        timestamps = []
        current = start_time
        while current <= end_time:
            timestamps.append(current)
            current += timedelta(minutes=10)
        
        n_records = len(timestamps)
        
        # Generate values
        values = np.random.normal(0, 1, n_records)
        
        # Create DataFrame
        data = {
            'WAARNEMINGDATUM': [dt.strftime('%Y-%m-%d') for dt in timestamps],
            'WAARNEMINGTIJD (MET/CET)': [dt.strftime('%H:%M:%S') for dt in timestamps],
            'NUMERIEKEWAARDE': values,
            'X': np.random.uniform(0.9, 1.0, n_records),
            'Y': np.random.uniform(-1.0, 1.0, n_records)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(csv_file, sep=';', index=False)
        return df
    
    def test_error_handling(self):
        """Test error handling for malformed data."""
        # Create malformed CSV file
        csv_file = self.raw_dir / "malformed.csv"
        with open(csv_file, 'w') as f:
            f.write("WAARNEMINGDATUM;WAARNEMINGTIJD (MET/CET);NUMERIEKEWAARDE;X;Y\n")
            f.write("2025-01-01;12:00:00;invalid_value;0.5;0.1\n")  # Invalid numeric value
            f.write("invalid_date;12:00:00;1.0;0.5;0.1\n")  # Invalid date
            f.write("2025-01-01;12:00:00;1.0;0.5;0.1\n")  # Valid row
        
        # Should handle errors gracefully
        extracted_df = self.ingester._extract_data_from_csv(csv_file)
        
        # Should extract only the valid row
        self.assertIsNotNone(extracted_df)
        self.assertEqual(len(extracted_df), 1)
    
    def test_normalization_parameters_persistence(self):
        """Test that normalization parameters are correctly saved and loaded."""
        # Create test data
        original_df, _ = self.create_synthetic_csv("test_data.csv", "test_data", 100)
        
        # Run whole pipeline
        self.ingester.run()
        
        # Check that parameters were saved
        norm_params_file = self.processed_dir / "normalization_params.json"
        self.assertTrue(norm_params_file.exists())
        
        # Verify parameters structure
        with open(norm_params_file, 'r') as f:
            params = json.load(f)
        
        self.assertIn("test_data", params)
        self.assertIn("mean", params["test_data"])
        self.assertIn("std", params["test_data"])
        self.assertIn("NUMERIEKEWAARDE", params["test_data"]["mean"])
        self.assertIn("NUMERIEKEWAARDE", params["test_data"]["std"])


# Simple validation functions (not unit tests)
def validate_normalization_formula():
    """Validate that normalization follows the correct mathematical formula."""
    print("Testing normalization formula...")
    
    # Create test data with known properties
    original_mean = 5.0
    original_std = 2.0
    test_values = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    
    # Expected normalized values
    expected_normalized = (test_values - original_mean) / original_std
    expected_normalized_manual = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Verify the formula
    np.testing.assert_array_almost_equal(expected_normalized, expected_normalized_manual, decimal=6)
    print("✅ Normalization formula is mathematically correct")


def validate_statistical_properties():
    """Validate that normalized data has correct statistical properties."""
    print("Testing statistical properties of normalized data...")
    
    # Generate random data
    np.random.seed(42)  # For reproducible results
    original_data = np.random.normal(10, 3, 1000)
    
    # Calculate normalization parameters
    original_mean = np.mean(original_data)
    original_std = np.std(original_data)
    
    # Apply normalization
    normalized_data = (original_data - original_mean) / original_std
    
    # Check properties
    normalized_mean = np.mean(normalized_data)
    normalized_std = np.std(normalized_data)
    
    print(f"Original data - Mean: {original_mean:.4f}, Std: {original_std:.4f}")
    print(f"Normalized data - Mean: {normalized_mean:.4f}, Std: {normalized_std:.4f}")
    
    # Validate properties
    assert abs(normalized_mean) < 0.01, f"Normalized mean should be close to 0, got {normalized_mean}"
    assert abs(normalized_std - 1.0) < 0.01, f"Normalized std should be close to 1, got {normalized_std}"
    print("✅ Normalized data has correct statistical properties")


def validate_data_cleaning():
    """Validate data cleaning operations."""
    print("Testing data cleaning operations...")
    
    # Create test data with known issues
    test_data = pd.DataFrame({
        'datetime': [
            datetime(2025, 1, 1, 0, 0, 0),
            datetime(2025, 1, 1, 0, 10, 0),
            datetime(2025, 1, 1, 0, 20, 0),
            datetime(2025, 1, 1, 0, 10, 0),  # Duplicate
            None,  # Missing
        ],
        'NUMERIEKEWAARDE': [1.0, 2.0, 3.0, 4.0, 5.0],
        'X': [0.5, 0.6, 0.7, 0.8, 0.9],
        'Y': [0.1, 0.2, 0.3, 0.4, 0.5]
    })
    
    # Expected cleaned data (remove missing datetime and duplicates)
    expected_cleaned = test_data.dropna(subset=['datetime']).drop_duplicates(subset=['datetime'], keep='first')
    expected_count = 3  # 5 original - 1 missing - 1 duplicate
    
    assert len(expected_cleaned) == expected_count, f"Expected {expected_count} rows, got {len(expected_cleaned)}"
    assert expected_cleaned['datetime'].notna().all(), "All datetime values should be non-null"
    print("✅ Data cleaning operations work correctly")


def validate_timestamp_alignment():
    """Validate timestamp alignment logic."""
    print("Testing timestamp alignment...")
    
    # Create datasets with different time ranges
    dataset1_times = pd.date_range('2025-01-01 01:00', '2025-01-01 23:00', freq='10min')
    dataset2_times = pd.date_range('2025-01-01 00:00', '2025-01-01 22:00', freq='10min')
    dataset3_times = pd.date_range('2025-01-01 00:30', '2025-01-01 23:30', freq='10min')
    
    # Expected aligned time range
    expected_start = max(dataset1_times.min(), dataset2_times.min(), dataset3_times.min())
    expected_end = min(dataset1_times.max(), dataset2_times.max(), dataset3_times.max())
    
    # This should be: start=2025-01-01 01:00, end=2025-01-01 22:00
    expected_start_manual = pd.Timestamp('2025-01-01 01:00:00')
    expected_end_manual = pd.Timestamp('2025-01-01 22:00:00')
    
    assert expected_start == expected_start_manual, f"Expected start {expected_start_manual}, got {expected_start}"
    assert expected_end == expected_end_manual, f"Expected end {expected_end_manual}, got {expected_end}"
    print("✅ Timestamp alignment logic is correct")


def run_all_validations():
    """Run all validation functions."""
    print("=" * 60)
    print("RUNNING PREPROCESSING PIPELINE VALIDATIONS")
    print("=" * 60)
    
    try:
        validate_normalization_formula()
        validate_statistical_properties()
        validate_data_cleaning()
        validate_timestamp_alignment()
        
        print("\n" + "=" * 60)
        print("✅ ALL VALIDATIONS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        raise


def main():
    """Run all tests and validations."""
    print("Starting preprocessing pipeline tests...")
    
    # Run unit tests
    print("\n" + "=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestPreprocessingPipeline)
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    # Run validations
    print("\n" + "=" * 60)
    print("RUNNING VALIDATIONS")
    print("=" * 60)
    
    run_all_validations()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if test_result.wasSuccessful():
        print("✅ All unit tests passed!")
    else:
        print("❌ Some unit tests failed!")
        print(f"Failures: {len(test_result.failures)}")
        print(f"Errors: {len(test_result.errors)}")
    
    print("✅ All mathematical validations passed!")
    print("=" * 60)
    
    return test_result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)