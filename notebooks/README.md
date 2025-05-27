# Notebooks

This directory contains Jupyter notebooks for exploratory data analysis, prototyping, and visualization.

## Guidelines

- Organize notebooks by topic or analysis type
- Use descriptive names (e.g., `01_data_exploration.ipynb`)
- Clear the output before committing
- Add a brief description at the top of each notebook
- Document dependencies and data sources

## Best Practices

1. **Structure**:
   - Start with a title and description
   - Include sections with clear headings
   - Add markdown cells to explain the purpose of each section

2. **Code Quality**:
   - Use functions and classes for reusable code
   - Add comments for complex logic
   - Keep cells focused on a single task

3. **Reproducibility**:
   - Specify package versions
   - Include random seeds
   - Document data loading steps

## Example Structure

```
notebooks/
├── 01_eda/
│   ├── 01_data_exploration.ipynb
│   └── 02_feature_analysis.ipynb
├── 02_modeling/
│   ├── 01_baseline_model.ipynb
│   └── 02_model_comparison.ipynb
└── utils/
    └── data_visualization.py
```

## Running Notebooks

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   jupyter lab  # or jupyter notebook
   ```

2. Start the Jupyter server:
   ```bash
   jupyter lab
   ```

3. Open the desired notebook in your browser.
