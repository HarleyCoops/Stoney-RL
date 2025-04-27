from huggingface_hub import HfApi

# Initialize the Hugging Face API
api = HfApi()

# File path
local_path = "C:\\Users\\chris\\Stoney-RL\\synthetic_stoney_data_fixed2.jsonl"
repo_id = "HarleyCooper/synthetic_stoney_data"
path_in_repo = "synthetic_stoney_data_fixed2.jsonl"

# First, create the repository if it doesn't exist
try:
    print(f"Creating repository {repo_id} if it doesn't exist...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True
    )
    print("Repository is ready.")
except Exception as e:
    print(f"Repository setup failed: {e}")
    exit(1)
    
# Upload the file
try:
    print(f"Uploading {local_path} to {repo_id}/{path_in_repo}...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset"
    )
    print("Upload successful!")
except Exception as e:
    print(f"Upload failed: {e}") 