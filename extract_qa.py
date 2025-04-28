import json
import sys
import os

def extract_qa_from_jsonl(input_filename, output_filename):
    """
    Reads a JSONL file line by line, extracts 'question' and 'answer' fields,
    and writes them to a new JSONL file. Handles JSON parsing errors.

    Args:
        input_filename (str): Path to the input JSONL file.
        output_filename (str): Path to the output JSONL file.
    """
    processed_lines = 0
    extracted_lines = 0
    error_lines = 0

    print(f"Starting extraction from '{input_filename}' to '{output_filename}'...")

    try:
        with open(input_filename, 'r', encoding='utf-8') as infile, \
             open(output_filename, 'w', encoding='utf-8') as outfile:

            for i, line in enumerate(infile):
                processed_lines = i + 1
                line = line.strip()
                if not line: # Skip empty lines
                    continue

                try:
                    # Attempt to parse the line as JSON
                    data = json.loads(line)

                    # Check if required keys exist
                    if "question" in data and "answer" in data:
                        # Create a new dictionary with only question and answer
                        qa_pair = {
                            "question": data["question"],
                            "answer": data["answer"]
                        }
                        # Write the new JSON object to the output file
                        outfile.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
                        extracted_lines += 1
                    else:
                        print(f"WARNING: Line {processed_lines}: Missing 'question' or 'answer' key. Skipping.", file=sys.stderr)
                        # Optionally print the line: print(f"Skipped line content: {line}", file=sys.stderr)
                        error_lines += 1

                except json.JSONDecodeError as e:
                    print(f"ERROR: Line {processed_lines}: Invalid JSON: {e}. Skipping.", file=sys.stderr)
                    # Optionally print the line: print(f"Skipped line content: {line}", file=sys.stderr)
                    error_lines += 1
                except Exception as e_generic:
                    print(f"ERROR: Line {processed_lines}: Unexpected error: {e_generic}. Skipping.", file=sys.stderr)
                    # Optionally print the line: print(f"Skipped line content: {line}", file=sys.stderr)
                    error_lines += 1

        print(f"\n--- Extraction Summary ---")
        print(f"Total lines processed: {processed_lines}")
        print(f"Lines successfully extracted (question/answer pairs): {extracted_lines}")
        print(f"Lines skipped due to errors or missing keys: {error_lines}")
        print(f"Output written to '{output_filename}'")
        print(f"------------------------\n")
        return True

    except FileNotFoundError:
        print(f"FATAL ERROR: Input file '{input_filename}' not found.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred during file processing: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    input_file = "OriginalXXX_synthetic_stoney_data.jsonl"
    # --- Choose your output filename ---
    output_file = "extracted_qa_pairs.jsonl" # Proposed name
    # output_file = "synthetic_stoney_qa_pairs.jsonl" # Alternative if you prefer this existing name

    if os.path.abspath(input_file) == os.path.abspath(output_file):
        print(f"FATAL ERROR: Input and output filenames cannot be the same ('{input_file}'). Choose a different output filename.", file=sys.stderr)
        sys.exit(1)

    success = extract_qa_from_jsonl(input_file, output_file)

    if not success:
        print("Extraction process failed.")
        sys.exit(1)
    else:
        print("Extraction process completed successfully.")
