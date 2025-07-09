"""
Data fetching script for RWS (Rijkswaterstaat) API endpoints.

This script fetches data from various RWS endpoints and stores it as CSV files
in the data/raw directory. It dynamically finds the earliest available data
for each endpoint.
"""

import requests
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any, Tuple
import urllib.parse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RWSDataFetcher:
    """Class to handle data fetching from RWS API endpoints."""
    
    def __init__(self):
        """Initialize the RWS data fetcher."""
        self.base_url = "https://rwsos.rws.nl/wb-api/dd/2.0/timeseries"
        self.session = requests.Session()
        
        # Create data directory if it doesn't exist
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RWS Data fetcher initialized. Data will be saved to {self.data_dir}")
    
    def test_date_range(self, location_code: str, observation_type_id: str, 
                       start_date: datetime, end_date: datetime,
                       source_name: str = "S_1") -> bool:
        """
        Test if a specific date range returns data.
        
        Args:
            location_code: Location code (e.g., 'SPY1', 'SPY')
            observation_type_id: Observation type ID (e.g., 'WT', 'SGA.3')
            start_date: Start date to test
            end_date: End date to test
            source_name: Source name (default: 'S_1')
            
        Returns:
            True if data is available, False otherwise
        """
        params = {
            'locationCode': location_code,
            'observationTypeId': observation_type_id,
            'sourceName': source_name,
            'startTime': start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'endTime': end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                # Check if there's actual data in the response
                if 'results' in data and len(data['results']) > 0:
                    events = data['results'][0].get('events', [])
                    return len(events) > 0
                return False
            else:
                return False
                
        except Exception as e:
            logger.debug(f"Test failed for {start_date} to {end_date}: {e}")
            return False
    
    def find_earliest_available_date(self, location_code: str, observation_type_id: str,
                                   source_name: str = "S_1") -> Optional[datetime]:
        """
        Find the earliest date for which data is available using binary search.
        
        Args:
            location_code: Location code
            observation_type_id: Observation type ID
            source_name: Source name
            
        Returns:
            Earliest available date or None if not found
        """
        logger.info(f"Finding earliest available date for {location_code}/{observation_type_id}")
        
        # Start with a reasonable range
        end_date = datetime.now()
        earliest_known = None
        
        # First, try some recent dates to establish a baseline
        test_dates = [
            datetime.now() - timedelta(days=30),
            datetime.now() - timedelta(days=90),
            datetime.now() - timedelta(days=365),
            datetime(2020, 1, 1),
            datetime(2015, 1, 1),
            datetime(2010, 1, 1)
        ]
        
        # Find the earliest date that works
        for test_date in test_dates:
            if self.test_date_range(location_code, observation_type_id, test_date, end_date, source_name):
                earliest_known = test_date
                logger.info(f"Found working date: {earliest_known}")
                break
            time.sleep(1)  # Be respectful to the API
        
        if earliest_known is None:
            logger.warning(f"No data found for {location_code}/{observation_type_id}")
            return None
        
        # Now do a more granular search around the earliest known date
        # Search month by month backwards
        current_date = earliest_known
        while current_date > datetime(2020, 1, 1):  # Don't go earlier than 2000
            # Try one month earlier
            test_date = current_date - timedelta(days=30)
            
            if self.test_date_range(location_code, observation_type_id, test_date, current_date, source_name):
                current_date = test_date
                logger.info(f"Found earlier data: {current_date}")
            else:
                # Try one week earlier
                test_date = current_date - timedelta(days=7)
                if self.test_date_range(location_code, observation_type_id, test_date, current_date, source_name):
                    current_date = test_date
                    logger.info(f"Found earlier data (week): {current_date}")
                else:
                    # Try one day earlier
                    test_date = current_date - timedelta(days=1)
                    if self.test_date_range(location_code, observation_type_id, test_date, current_date, source_name):
                        current_date = test_date
                        logger.info(f"Found earlier data (day): {current_date}")
                    else:
                        break
            
            time.sleep(1)  # Be respectful to the API
        
        logger.info(f"Earliest available date for {location_code}/{observation_type_id}: {current_date}")
        return current_date
    
    def fetch_data_with_date_range(self, location_code: str, observation_type_id: str,
                                  start_date: datetime, end_date: datetime,
                                  source_name: str = "S_1") -> Optional[Dict]:
        """
        Fetch data for a specific date range.
        
        Args:
            location_code: Location code
            observation_type_id: Observation type ID
            start_date: Start date
            end_date: End date
            source_name: Source name
            
        Returns:
            Response data as dictionary or None if failed
        """
        params = {
            'locationCode': location_code,
            'observationTypeId': observation_type_id,
            'sourceName': source_name,
            'startTime': start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'endTime': end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
        try:
            logger.info(f"Fetching data for {location_code}/{observation_type_id} from {start_date} to {end_date}")
            response = self.session.get(self.base_url, params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully fetched data for {location_code}/{observation_type_id}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data for {location_code}/{observation_type_id}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for {location_code}/{observation_type_id}: {e}")
            return None
    
    def parse_rws_data(self, data: Dict) -> pd.DataFrame:
        """
        Parse RWS API response data into a pandas DataFrame.
        
        Args:
            data: Response data from RWS API
            
        Returns:
            Parsed DataFrame with columns: datetime, datetime_unix, NUMERIEKEWAARDE, X, Y
        """
        try:
            # Extract the time series data
            if 'results' in data and len(data['results']) > 0:
                time_series = data['results'][0]
                
                # Extract events (the actual data points)
                events = time_series.get('events', [])
                
                if not events:
                    logger.warning("No events found in the response")
                    return pd.DataFrame()
                
                # Parse events into DataFrame
                parsed_data = []
                for event in events:
                    try:
                        # Parse timestamp
                        timestamp_str = event.get('timeStamp', '')
                        if timestamp_str:
                            # Get numeric value
                            value = event.get('value', None)
                            
                            # Get coordinates if available
                            x = time_series.get('location', {}).get('geometry', {}).get('coordinates', [None])[0]
                            y = time_series.get('location', {}).get('geometry', {}).get('coordinates', [None])[1]
                            
                            parsed_data.append({
                                'timeStamp': timestamp_str,
                                'value': value,
                                'X': x,
                                'Y': y
                            })
                    except Exception as e:
                        logger.warning(f"Failed to parse event: {e}")
                        continue
                
                df = pd.DataFrame(parsed_data)
                logger.info(f"Parsed {len(df)} data points")
                return df
                
            else:
                logger.error("No timeSeries found in response")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to parse RWS data: {e}")
            return pd.DataFrame()
    
    def save_to_csv(self, df: pd.DataFrame, filename: str) -> bool:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Name of the CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if df.empty:
                logger.warning(f"No data to save for {filename}")
                return False
            
            filepath = self.data_dir / filename
            df.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath} ({len(df)} rows)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save data to {filename}: {e}")
            return False
    
    def fetch_all_endpoints(self):
        """Fetch data from all specified RWS endpoints."""
        
        # Define all endpoints to fetch
        endpoints = {
            "water_height": {
                "location_code": "SPY1",
                "observation_type_id": "WT",
                "filename": "water_height.csv"
            },
            "cross_current": {
                "location_code": "SPY1", 
                "observation_type_id": "SGA.3",
                "filename": "cross_current.csv"
            },
            "wind_speed": {
                "location_code": "SPY1",
                "observation_type_id": "WN.4", 
                "filename": "wind_speed.csv"
            },
            "wind_direction": {
                "location_code": "SPY1",
                "observation_type_id": "WN.2",
                "filename": "wind_direction.csv"
            },
            "wave_height": {
                "location_code": "SPY",
                "observation_type_id": "GH10.1",
                "filename": "wave_height.csv"
            }
        }
        
        # Fetch data from each endpoint
        for data_type, config in endpoints.items():
            logger.info(f"Processing {data_type}...")
            
            try:
                # Find the earliest available date for this endpoint
                # earliest_date = self.find_earliest_available_date(
                #     location_code=config["location_code"],
                #     observation_type_id=config["observation_type_id"]
                # )
                earliest_date = datetime(2025, 1, 1)
                
                if earliest_date is None:
                    logger.error(f"Could not find earliest date for {data_type}")
                    continue
                
                # Fetch data from earliest date to now
                end_date = datetime.now()
                data = self.fetch_data_with_date_range(
                    location_code=config["location_code"],
                    observation_type_id=config["observation_type_id"],
                    start_date=earliest_date,
                    end_date=end_date
                )
                
                if data:
                    # Parse the data
                    df = self.parse_rws_data(data)
                    
                    if not df.empty:
                        # Save to CSV
                        success = self.save_to_csv(df, config["filename"])
                        
                        if success:
                            logger.info(f"Successfully fetched and saved {data_type} data from {earliest_date} to {end_date}")
                        else:
                            logger.error(f"Failed to save {data_type} data")
                    else:
                        logger.warning(f"No data parsed for {data_type}")
                else:
                    logger.error(f"No data received for {data_type}")
                    
            except Exception as e:
                logger.error(f"Error processing {data_type}: {e}")
            
            # Add delay between endpoints to be respectful to the API
            time.sleep(3)
        
        logger.info("Data fetching completed!")


def main():
    """Main function to fetch all required data."""
    
    # Initialize fetcher
    fetcher = RWSDataFetcher()
    
    # Fetch all data
    fetcher.fetch_all_endpoints()


if __name__ == "__main__":
    main()