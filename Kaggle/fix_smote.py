import json
import re

# Load the notebook
notebook_path = 'Kaggle/Extended_Solution.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find cells with SMOTE applied to the entire dataset
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Check for incorrect SMOTE implementation (applied to entire dataset)
        if 'smt.fit_sample(X, y)' in source and 'train_input_new, train_output_new =' in source:
            print(f"Found incorrect SMOTE implementation in cell {i}")
            
            # Create new correct implementation
            new_code = [
                "# First split the data into train and test sets\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)\n",
                "\n",
                "# Print original training data distribution\n",
                "print('Original training dataset shape {}'.format(Counter(y_train)))\n",
                "\n",
                "# Apply SMOTE only to the training data\n",
                "smt = SMOTE(random_state=20)\n",
                "X_train_resampled, y_train_resampled = smt.fit_sample(X_train, y_train)\n",
                "\n",
                "# Print new training data distribution\n",
                "print('New training dataset shape {}'.format(Counter(y_train_resampled)))\n",
                "\n",
                "# Convert to DataFrame with original column names\n",
                "X_train_resampled = pd.DataFrame(X_train_resampled, columns=list(X.columns))\n"
            ]
            
            # Replace the cell content
            notebook['cells'][i]['source'] = new_code
            
            # Now find the train-test split applied after SMOTE, which needs to be removed
            for j, next_cell in enumerate(notebook['cells'][i+1:], start=i+1):
                if next_cell['cell_type'] == 'code' and 'source' in next_cell:
                    next_source = ''.join(next_cell['source']) if isinstance(next_cell['source'], list) else next_cell['source']
                    
                    # Look for train_test_split on train_input_new
                    if 'train_test_split(train_input_new, train_output_new' in next_source:
                        print(f"Found incorrect train-test split in cell {j}")
                        # Replace with comment explaining the change
                        notebook['cells'][j]['source'] = [
                            "# NOTE: The train-test split has already been performed before SMOTE.\n",
                            "# We'll use X_train_resampled and y_train_resampled for model training,\n",
                            "# and X_test, y_test for evaluation.\n"
                        ]
            
            # Also find model training cells and update them to use resampled data
            for j, model_cell in enumerate(notebook['cells'][i+1:], start=i+1):
                if model_cell['cell_type'] == 'code' and 'source' in model_cell:
                    model_source = ''.join(model_cell['source']) if isinstance(model_cell['source'], list) else model_cell['source']
                    
                    # Look for model.fit(X_train, y_train)
                    if re.search(r'\.fit\(X_train, y_train\)', model_source):
                        print(f"Updating model training in cell {j}")
                        # Replace with resampled data
                        new_source = re.sub(
                            r'\.fit\(X_train, y_train\)', 
                            '.fit(X_train_resampled, y_train_resampled)', 
                            model_source
                        )
                        if isinstance(model_cell['source'], list):
                            notebook['cells'][j]['source'] = [new_source]
                        else:
                            notebook['cells'][j]['source'] = new_source

# Save the modified notebook
with open(notebook_path + '.fixed', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"Fixed notebook saved to {notebook_path}.fixed") 