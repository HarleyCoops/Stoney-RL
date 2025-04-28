import json
import sys

def process_jsonl(input_filename, output_filename):
    """
    Reads a large JSONL file line by line, extracts 'question' and 'answer' 
    fields, and writes them to a new JSONL file.

    Args:
        input_filename (str): The path to the input JSONL file.
        output_filename (str): The path to the output JSONL file.
    """
    processed_count = 0
    try:
        with open(input_filename, 'r', encoding='utf-8') as infile, \
             open(output_filename, 'w', encoding='utf-8') as outfile:
            for line in infile:
                try:
                    # Attempt to parse the line as JSON
                    data = json.loads(line.strip())
                    
                    # Check if 'question' and 'answer' keys exist
                    if 'question' in data and 'answer' in data:
                        # Create a new dictionary with only 'question' and 'answer'
                        qa_pair = {
                            'question': data['question'],
                            'answer': data['answer']
                        }
                        # Write the new JSON object to the output file as a line
                        outfile.write(json.dumps(qa_pair, ensure_ascii=False) + '\\n')
                        processed_count += 1
                    else:
                        print(f"Skipping line due to missing 'question' or 'answer': {line.strip()}", file=sys.stderr)

                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}", file=sys.stderr)
                except Exception as e:
                    print(f"An unexpected error occurred processing line: {line.strip()} - Error: {e}", file=sys.stderr)

        print(f"Successfully processed {processed_count} lines.")
        print(f"Output written to {output_filename}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    input_file = "OriginalXXX_synthetic_stoney_data.jsonl"
    output_file = "synthetic_stoney_qa_pairs.jsonl"
    
    process_jsonl(input_file, output_file) 