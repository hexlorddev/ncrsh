# Processed Data

This directory contains processed and cleaned data files ready for analysis and model training.

## Guidelines

- Store cleaned and processed versions of raw data here
- Include a version number in filenames (e.g., `processed_data_v1.parquet`)
- Document the processing steps in a README or separate documentation
- Keep the directory organized by dataset and version
- Include a data dictionary for processed files

## Processing Steps

For each processed dataset, document:

1. Source raw data file
2. Cleaning steps applied
3. Transformations performed
4. Any feature engineering
5. Train/test/validation splits

## Example Structure

```
processed/
├── dataset1/
│   ├── README.md
│   ├── dataset1_processed_v1.parquet
│   └── dataset1_features_v1.pkl
└── dataset2/
    ├── README.md
    └── dataset2_processed_v1.h5
```
