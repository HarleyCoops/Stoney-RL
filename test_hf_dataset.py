from datasets import load_dataset

def test_load_dataset(file_path):
    """Test loading a JSONL file as a HuggingFace Dataset."""
    print(f"Attempting to load dataset from {file_path}...")
    
    try:
        # Load the dataset
        dataset = load_dataset('json', data_files=file_path)
        
        # Get basic info
        print(f"Dataset loaded successfully:")
        print(f"- Split names: {dataset.keys()}")
        print(f"- Number of examples: {len(dataset['train'])}")
        print(f"- Features: {dataset['train'].features}")
        
        # Show a sample
        print("\nSample entry:")
        print(dataset['train'][0])
        
        return True
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return False

if __name__ == "__main__":
    # Try loading the fixed dataset
    test_load_dataset('synthetic_stoney_data_fixed2.jsonl') 