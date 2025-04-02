import json
import re

# Load the notebook
notebook_path = 'Kaggle/Extended_Solution.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Update all instances of fit_sample to fit_resample
changes_made = 0
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Check if fit_sample is in the source
        if 'fit_sample' in source:
            print(f"Found fit_sample in cell {i}")
            changes_made += 1
            
            # Replace fit_sample with fit_resample
            new_source = source.replace('fit_sample', 'fit_resample')
            
            # Update the cell
            if isinstance(cell['source'], list):
                notebook['cells'][i]['source'] = [new_source]
            else:
                notebook['cells'][i]['source'] = new_source

# Save the updated notebook if changes were made
if changes_made > 0:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print(f"Updated {changes_made} instances of fit_sample to fit_resample in {notebook_path}")
else:
    print("No instances of fit_sample found in the notebook.") 