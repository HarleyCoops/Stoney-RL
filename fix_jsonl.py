import json
import os

def fix_jsonl_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                
                # For lines 36651 and beyond, check if entry_id exists and remove it
                if i >= 36651 and "entry_id" in data:
                    # Create a new dict with only the fields we want to keep
                    clean_data = {}
                    for key, value in data.items():
                        clean_data[key] = value
                        # Stop after processing original_id
                        if key == "original_id":
                            break
                    
                    # Write the cleaned data
                    outfile.write(json.dumps(clean_data) + '\n')
                else:
                    # For other lines, write them as is
                    outfile.write(line)
                    
            except json.JSONDecodeError:
                print(f"Error parsing line {i}: {line[:100]}...")
                # Write the line as is if there's a parsing error
                outfile.write(line)
    
    print(f"Fixed JSONL file created at {output_file}")

if __name__ == "__main__":
    input_file = "synthetic_stoney_data_fixed2.jsonl"
    output_file = "synthetic_stoney_data_fixed3.jsonl"
    
    if os.path.exists(input_file):
        fix_jsonl_file(input_file, output_file)
    else:
        print(f"Error: Input file {input_file} not found") 