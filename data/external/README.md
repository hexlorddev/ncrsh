# External Data

This directory contains data from external sources that are used in the project.

## Guidelines

- Store all third-party datasets here
- Include the original data files without modifications
- Add a README.md for each external dataset with:
  - Source URL
  - Date downloaded
  - License information
  - Any relevant notes about the data
- Keep the original directory structure if the data comes in multiple files

## Example Structure

```
external/
├── imagenet/
│   ├── README.md
│   └── imagenet_2012_bounding_boxes.csv
└── glove/
    ├── README.md
    ├── glove.6B.50d.txt
    └── glove.6B.100d.txt
```

## Adding New External Data

1. Create a new subdirectory with a descriptive name
2. Add the data files
3. Create a README.md with metadata
4. Update this README with a brief description of the new data
