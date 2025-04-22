import pandas as pd
import os
import papermill as pm

# File paths
csv_file = "./job_queries/success.csv"  # Replace with your CSV file path
notebook_path = "step1-featurizing.ipynb"  # Notebook path
output_dir = "./executed_notebooks"  # Directory to save executed notebooks
tensors_dir = "./tensors"  # Directory to save tensor files (add this if not defined)

# Ensure files exist
if not os.path.isfile(csv_file):
    raise FileNotFoundError(f"CSV file not found: {csv_file}")
if not os.path.isfile(notebook_path):
    raise FileNotFoundError(f"Notebook file not found: {notebook_path}")

# Ensure the output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(tensors_dir, exist_ok=True)

# Read the CSV file
data = pd.read_csv(csv_file)

# Validate columns
if "QUERYID" not in data.columns or "EXPLAIN_TIME" not in data.columns:
    raise KeyError("The CSV file must contain 'QUERYID' and 'EXPLAIN_TIME' columns.")

# Start processing from line 1727 (index 1726 since index is 0-based)
start_index = 1725  # Adjust to 0-based index
data = data.iloc[start_index:]

# Iterate over each row in the sliced DataFrame
for index, row in data.iterrows():
    try:
        # Extract QUERYID and EXPLAIN_TIME
        query_id = int(row["QUERYID"])  # Ensure QUERYID is an integer
        query1_ts = row["EXPLAIN_TIME"]  # EXPLAIN_TIME is likely a string or datetime

        # Check if tensor file already exists
        tensor_file = os.path.join(tensors_dir, f"QUERYID_{query_id}_query1_ts.pt")
        if os.path.isfile(tensor_file):
            print(f"Tensor file already exists for QUERYID: {query_id}, skipping tensor generation...")
            continue

        # Check if output notebook already exists
        output_notebook = os.path.join(output_dir, f"executed_notebook_{query_id}.ipynb")
        if os.path.isfile(output_notebook):
            print(f"Notebook already exists for QUERYID: {query_id}, skipping notebook execution...")
            continue

        # Pass parameters to the notebook and execute it
        print(f"Running notebook for QUERYID: {query_id}, query1_ts: {query1_ts}")
        pm.execute_notebook(
            input_path=notebook_path,
            output_path=output_notebook,
            parameters={
                "QUERYID": query_id,
                "query1_ts": query1_ts  # Pass EXPLAIN_TIME as query1_ts
            }
        )
    except Exception as e:
        print(f"Error executing notebook for QUERYID: {query_id}, query1_ts: {query1_ts}. Error: {e}")

print("Resumed execution and completed all notebooks successfully!")