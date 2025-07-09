"""
Data ingestion and preprocessing script for IJmuiden cross-current data.

This script processes raw CSV files containing various environmental measurements
and extracts, cleans, and normalizes the data for further analysis.
"""

import pandas as pd
import os
import json
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    current = Path.cwd()
    if current.name != "ijmuiden_cross-current":
        # Try to find the project directory
        for parent in current.parents:
            if (parent / "ijmuiden_cross-current").exists():
                os.chdir(parent / "ijmuiden_cross-current")
                break
        else:
            # If not found, try relative to current
            if (current / "ijmuiden_cross-current").exists():
                os.chdir(current / "ijmuiden_cross-current")
    
    print(f"Working directory: {Path.cwd()}")

class DataIngester:
    """Class to handle data ingestion and preprocessing."""
    
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        """
        Initialize the DataIngester.
        
        Args:
            raw_data_dir: Directory containing raw CSV files
            processed_data_dir: Directory to save processed data
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.relevant_columns = {
            'value': float,
            'timeStamp': str,
            'X': float,
            'Y': float,
        }
        self.date_col = 'timeStamp'
        self.value_col = 'value'

        self.norm_params = {}

    def _get_normalization_params(self, extracted_df: pd.DataFrame, data_type: str) -> dict[str, dict[str, float]]:
        """
        Calculate normalization parameters for a given data type.
        
        Args:
            extracted_df: DataFrame with extracted data
            data_type: Type of data being processed
            
        Returns:
            Dictionary with normalization parameters
        """
        if extracted_df is None or extracted_df.empty:
            logger.warning(f"No data available to calculate normalization parameters for {data_type}")
            return {}
        
        # Calculate statistics for each column
        params = {
            'mean': {},
            'std': {}
        }
        
        # Get numeric columns from relevant_columns
        numeric_columns = [
            column_name for column_name, column_type in self.relevant_columns.items()
            if column_type in [int, float]
        ]
        
        # Add datetime_unix as it's also numeric
        numeric_columns.append('datetime_unix')
        
        # Calculate parameters for each numeric column
        for column_name in numeric_columns:
            if column_name in extracted_df.columns:
                valid_values = extracted_df[column_name].dropna()
                if len(valid_values) > 0:
                    params['mean'][column_name] = float(valid_values.mean())
                    params['std'][column_name] = float(valid_values.std())
                    logger.debug(f"Calculated params for {column_name}: mean={params['mean'][column_name]:.2f}, std={params['std'][column_name]:.2f}")
        
        # Store the parameters in the instance variable
        self.norm_params[data_type] = params
        
        logger.info(f"Calculated normalization parameters for {data_type}")
        return params
    
    def _load_normalization_params(self) -> Dict:
        """Load normalization parameters from JSON file."""
        norm_params_file = self.raw_data_dir / "norm_params.json"
        if norm_params_file.exists():
            with open(norm_params_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning("Normalization parameters file not found. Using default values.")
            return {}
    
    def _save_normalization_params(self):
        """Save normalization parameters to JSON file."""
        norm_params_file = self.processed_data_dir / "normalization_params.json"
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_params = {}
        for data_type, params in self.norm_params.items():
            serializable_params[data_type] = {
                'mean': {k: float(v) if v is not None else None for k, v in params['mean'].items()},
                'std': {k: float(v) if v is not None else None for k, v in params['std'].items()}
            }
        
        with open(norm_params_file, 'w') as f:
            json.dump(serializable_params, f, indent=2)
        
        logger.info(f"Saved normalization parameters to {norm_params_file}")
    
    def _parse_datetime(self, date_str: str) -> Optional[datetime]:
        """
        Parse datetime string into datetime object.
        Handles various formats including ISO 8601 with 'Z' timezone.
        
        Args:
            date_str: Date string to parse
            
        Returns:
            datetime object or None if parsing fails
        """
        try:
            # Handle ISO 8601 format with 'Z' timezone (UTC)
            if 'T' in date_str and 'Z' in date_str:
                # Remove 'Z' and parse as UTC
                dt_str = date_str.replace('Z', '+00:00')
                return datetime.fromisoformat(dt_str)
            
            # Handle other formats (your existing logic)
            elif '-' in date_str:
                # Handle YYYY-MM-DD HH:MM:SS format
                if len(date_str.split('-')[0]) == 4:  # YYYY-MM-DD format
                    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                else:  # DD-MM-YYYY format
                    return datetime.strptime(date_str, "%d-%m-%Y %H:%M:%S")
            else:
                return datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")
                
        except Exception as e:
            logger.warning(f"Could not parse datetime: {date_str}. Error: {e}")
            return None
    
    def _extract_data_from_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Extract specific data type from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with extracted data or None if extraction fails
        """
        try:
            logger.info(f"Processing {file_path}")
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Extract relevant columns
            result_data = []
            
            for _, row in df.iterrows():
                try:
                    # Initialize row data
                    row_data = {}
                    
                    # Extract datetime first (special handling)
                    if self.date_col in df.columns:
                        dt = self._parse_datetime(str(row[self.date_col]))
                        # skip for invalid datetime format
                        if dt is None:
                            continue

                        row_data['datetime'] = dt
                        row_data['datetime_unix'] = dt.timestamp() if dt else None
                    else:
                        continue  # Skip rows without datetime
                    
                    # Extract other columns based on their types
                    for column_name, column_type in self.relevant_columns.items():
                        if column_name in [self.date_col]:  # Skip date/time as they're handled above
                            continue
                            
                        if column_name in df.columns and pd.notna(row[column_name]):
                            try:
                                raw_value = row[column_name]
                                
                                # Handle different data types
                                if column_type == str:
                                    value = str(raw_value)
                                elif column_type == int:
                                    # Handle potential decimal values for int conversion
                                    if isinstance(raw_value, str):
                                        value = int(float(raw_value.replace(',', '.')))
                                    else:
                                        value = int(float(raw_value))
                                elif column_type == float:
                                    # Handle potential comma decimal separators
                                    if isinstance(raw_value, str):
                                        value = float(raw_value.replace(',', '.'))
                                    else:
                                        value = float(raw_value)
                                else:
                                    value = column_type(raw_value)
                                
                                # Skip invalid values for numeric types
                                if column_type in [int, float] and value in [-999999999, 999999999]:
                                    continue
                                
                                # Use the original column name
                                row_data[column_name] = value
                                    
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Could not convert {column_name} to {column_type.__name__}: {raw_value}. Error: {e}")
                                if column_name == self.value_col:  # Skip rows with invalid main values
                                    continue
                                else:  # For optional columns like coordinates, set to None
                                    row_data[column_name] = None

                        # if column_name is not in df.columns, we do the following:
                        else:
                            # Handle missing columns
                            if column_name == self.value_col:  # Skip rows without main value
                                continue
                            else:  # For optional columns, set to None
                                row_data[column_name] = None
                    
                    # Only add row if we have the essential data (datetime and value)
                    if 'datetime' in row_data and self.value_col in row_data:
                        result_data.append(row_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing row: {e}")
                    continue
            
            if not result_data:
                logger.warning(f"No valid data extracted from {file_path}")
                return None
            
            result_df = pd.DataFrame(result_data)
            
            # Sort by datetime
            result_df = result_df.sort_values('datetime')
            
            logger.info(f"Extracted {len(result_df)} records from {file_path}")
            return result_df
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def _normalize_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Normalize data using the provided normalization parameters.
        
        Args:
            df: DataFrame to normalize
            data_type: Type of data being normalized
            
        Returns:
            Normalized DataFrame
        """
        if data_type not in self.norm_params:
            logger.warning(f"No normalization parameters found for {data_type}")
            return df
        
        params = self.norm_params[data_type]
        normalized_df = df.copy()
        
        # Get numeric columns from relevant_columns
        numeric_columns = [
            column_name for column_name, column_type in self.relevant_columns.items()
            if column_type in [int, float]
        ]
        
        # Add datetime_unix as it's also numeric
        numeric_columns.append('datetime_unix')
        
        # Normalize each numeric column
        for column_name in numeric_columns:
            if (column_name in params['mean'] and 
                column_name in params['std'] and 
                params['std'][column_name] is not None and 
                params['std'][column_name] != 0 and
                column_name in df.columns):
                
                mean_val = params['mean'][column_name]
                std_val = params['std'][column_name]
                
                normalized_df[column_name] = (df[column_name] - mean_val) / std_val
                logger.debug(f"Normalized {column_name} using mean={mean_val:.2f}, std={std_val:.2f}")
        
        return normalized_df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by removing outliers and invalid values.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        cleaned_df = df.copy()
        
        # Remove rows with missing datetime
        cleaned_df = cleaned_df.dropna(subset=['datetime'])
        cleaned_df = cleaned_df.dropna(subset=[self.value_col])
        
        # Remove extreme outliers (values beyond 3 standard deviations)
        # if len(cleaned_df) > 10:
        #     mean_val = cleaned_df['NUMERIEKEWAARDE'].mean()
        #     std_val = cleaned_df['NUMERIEKEWAARDE'].std()
        #     lower_bound = mean_val - 3 * std_val
        #     upper_bound = mean_val + 3 * std_val
        #     cleaned_df = cleaned_df[
        #         (cleaned_df['NUMERIEKEWAARDE'] >= lower_bound) & 
        #         (cleaned_df['NUMERIEKEWAARDE'] <= upper_bound)
        #     ]
        
        # Remove duplicate timestamps (keep the first occurrence)
        cleaned_df = cleaned_df.drop_duplicates(subset=['datetime'], keep='first')
        
        return cleaned_df
    
    def process_all_files(self) -> Dict[str, pd.DataFrame]:
        """
        Process all CSV files in the raw data directory.
        
        Returns:
            Dictionary mapping data types to processed DataFrames
        """
        logger.info("Starting data processing...")
        
        # Get all CSV files
        csv_files = list(self.raw_data_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        # Dictionary to store processed data for each type
        processed_data = {}

        # min max_timestamp, max min_timestamp
        max_timestamp = []
        min_timestamp = []
        
        # Process each file
        for file_path in csv_files:
            try:
                # Extract data
                data_type = file_path.stem
                extracted_df = self._extract_data_from_csv(file_path)
                
                if extracted_df is not None and not extracted_df.empty:
                    # Clean the data
                    cleaned_df = self._clean_data(extracted_df)
                    self._get_normalization_params(cleaned_df, data_type)

                    max_timestamp.append(cleaned_df['datetime'].max())
                    min_timestamp.append(cleaned_df['datetime'].min())
                    
                    if not cleaned_df.empty:
                        # Normalize the data
                        normalized_df = self._normalize_data(cleaned_df, data_type)
                        processed_data[data_type] = (normalized_df)
                        
            except Exception as e:
                logger.error(f"Error processing {data_type} from {file_path}: {e}")
                continue
        
        # Combine all data for each type (this is assuming multiple files for the same data type)
        # final_data = {}
        # for data_type, dataframes in processed_data.items():
        #     if dataframes:
        #         combined_df = pd.concat(dataframes, ignore_index=True)
        #         combined_df = combined_df.sort_values('datetime')
        #         final_data[data_type] = combined_df
        #         logger.info(f"Final dataset for {data_type}: {len(combined_df)} records")
        #     else:
        #         logger.warning(f"No data found for {data_type}")

        # align the timestamps such that no one data type has more data than the other.
        # we can either extend the data to the min max_timestamp or truncate the data to the smallest min_timestamp. for now, we truncate.
        for data_type, df in processed_data.items():
            if df is not None and not df.empty:
                # keep data between max min_timestamp and min max_timestamp
                df = df[(df['datetime'] >= max(min_timestamp)) & (df['datetime'] <= min(max_timestamp))]
                processed_data[data_type] = df

        return processed_data
    
    def save_processed_data(self, processed_data: Dict[str, pd.DataFrame]):
        """
        Save processed data to CSV files.
        
        Args:
            processed_data: Dictionary mapping data types to DataFrames
        """
        logger.info("Saving processed data...")
        
        for data_type, df in processed_data.items():
            if df is not None and not df.empty:
                output_file = self.processed_data_dir / f"{data_type}_processed.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Saved {data_type} data to {output_file} ({len(df)} records)")
    
    def run(self):
        """Main method to run the complete data ingestion pipeline."""
        logger.info("Starting data ingestion pipeline...")
        
        try:
            # Process all files
            processed_data = self.process_all_files()
            
            # Save processed data
            self.save_processed_data(processed_data)

            # Save normalization parameters
            self._save_normalization_params()
            
            logger.info("Data ingestion pipeline completed successfully!")
            
            # Print summary
            print("\n" + "="*50)
            print("DATA INGESTION SUMMARY")
            print("="*50)
            for data_type, df in processed_data.items():
                if df is not None and not df.empty:
                    print(f"{data_type}: {len(df)} records")
                    print(f"min timestamp: {df['datetime'].min()}")
                    print(f"max timestamp: {df['datetime'].max()}")
                    print(f"max value: {df['value'].max()}")
                    print(f"min value: {df['value'].min()}")
                else:
                    print(f"{data_type}: No data found")
            print("="*50)
            
        except Exception as e:
            logger.error(f"Error in data ingestion pipeline: {e}")
            raise


def main():
    """Main function to run the data ingestion."""
    setup_environment()
    ingester = DataIngester()
    ingester.run()


if __name__ == "__main__":
    main()
