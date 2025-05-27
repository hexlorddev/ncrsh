# Data Directory

This directory contains all data-related files for the ncrsh project.

## Directory Structure

- `raw/`: Raw data files (immutable)
- `processed/`: Processed and cleaned data
- `external/`: External data sources

## Usage

1. Place raw data files in the `raw/` directory
2. Processed data should be saved in the `processed/` directory
3. External datasets should be placed in the `external/` directory

## Data Versioning

All data files should be versioned. Use the following naming convention:
- Raw data: `raw/dataset_name_v{version}.ext`
- Processed data: `processed/processed_name_v{version}.ext`

## Adding New Data

1. Update this README with a description of the new data
2. Document the data source and any preprocessing steps
3. Add the data to the appropriate directory

## Data Processing Pipeline

1. Raw data is loaded from `raw/`
2. Processing scripts in `scripts/process_data.py` transform the data
3. Processed data is saved to `processed/`
4. Processed data is used for training and evaluation
