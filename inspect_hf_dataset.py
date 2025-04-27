from datasets import load_dataset
from pprint import pprint

# Load the dataset from Hugging Face Hub
ds = load_dataset("HarleyCooper/synthetic-stoney-data", split="train")

# Print dataset info
print("Dataset info:")
print(ds)

# Print column names
print("\nColumn names:")
print(ds.column_names)

# Print the first 3 examples
print("\nFirst 3 examples:")
pprint(ds[:3])

# Check for missing values in each column
print("\nMissing values per column:")
for col in ds.column_names:
    missing = sum(x is None for x in ds[col])
    print(f"{col}: {missing} missing values")