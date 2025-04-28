import json
import sys
import os

def fix_jsonl(input_filename, output_filename):
    """
    Reads a JSONL file, attempts to fix lines with JSON errors 
    (specifically unescaped newlines within strings), and writes 
    the processed lines to a new file.

    Args:
        input_filename (str): Path to the input JSONL file.
        output_filename (str): Path to the output JSONL file.
    """
    fixed_count = 0
    error_count = 0
    total_lines = 0
    
    try:
        with open(input_filename, 'r', encoding='utf-8') as infile, \
             open(output_filename, 'w', encoding='utf-8') as outfile:
            
            for i, line in enumerate(infile):
                total_lines = i + 1
                line = line.strip()
                if not line: # Skip empty lines
                    continue
                    
                try:
                    # Try parsing the original line
                    data = json.loads(line)
                    # If successful, write it back (ensuring proper JSON format)
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                except json.JSONDecodeError:
                    # If parsing fails, try fixing common issues (like literal newlines)
                    try:
                        # Replace likely problematic literal newlines with escaped ones
                        fixed_line = line.replace('\n', '\\n')
                        
                        # Try parsing the fixed line
                        data = json.loads(fixed_line)
                        # If successful, write the fixed data
                        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                        fixed_count += 1
                    except json.JSONDecodeError as e2:
                        # If fixing also fails, log the error and write the original line
                        print(f"ERROR: Line {total_lines}: Still invalid after attempting fix: {e2}. Writing original line.", file=sys.stderr)
                        print(f"Original line: {line}", file=sys.stderr)
                        outfile.write(line + '\n') # Write original problematic line
                        error_count += 1
                    except Exception as e_fix:
                        print(f"ERROR: Line {total_lines}: Unexpected error during fixing attempt: {e_fix}. Writing original line.", file=sys.stderr)
                        print(f"Original line: {line}", file=sys.stderr)
                        outfile.write(line + '\n') # Write original problematic line
                        error_count += 1
                except Exception as e_outer:
                    print(f"ERROR: Line {total_lines}: Unexpected error during initial processing: {e_outer}. Writing original line.", file=sys.stderr)
                    print(f"Original line: {line}", file=sys.stderr)
                    outfile.write(line + '\n') # Write original problematic line
                    error_count += 1

        print(f"\n--- Fixing Summary ---")
        print(f"Total lines processed: {total_lines}")
        print(f"Lines successfully fixed: {fixed_count}")
        print(f"Lines that remained invalid after fix attempt: {error_count}")
        print(f"Output written to {output_filename}")
        print(f"----------------------\n")
        return True

    except FileNotFoundError:
        print(f"FATAL ERROR: Input file '{input_filename}' not found.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred during file processing: {e}", file=sys.stderr)
        return False

def replace_file(source, destination):
    """ Safely replaces the destination file with the source file. """
    try:
        # Optional: Create a backup of the original file
        # backup_name = destination + ".bak"
        # print(f"Creating backup: {backup_name}")
        # os.replace(destination, backup_name) # Use replace for atomicity if possible
        
        print(f"Replacing '{destination}' with '{source}'...")
        os.replace(source, destination)
        print("Replacement successful.")
        return True
    except OSError as e:
        print(f"ERROR: Could not replace file '{destination}'. Error: {e}", file=sys.stderr)
        # Attempt to restore backup if it exists
        # if os.path.exists(backup_name):
        #     try:
        #         os.replace(backup_name, destination)
        #         print(f"Restored backup '{backup_name}' to '{destination}'.")
        #     except OSError as e_restore:
        #         print(f"CRITICAL ERROR: Failed to restore backup for '{destination}'. Error: {e_restore}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during file replacement: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    input_file = "OriginalXXX_synthetic_stoney_data.jsonl"
    temp_output_file = "temp_fixed_data.jsonl" # Use a temporary name
    
    print(f"Starting JSONL fixing process for '{input_file}'...")
    fix_successful = fix_jsonl(input_file, temp_output_file)
    
    if fix_successful:
        print("Fixing process completed. Attempting to replace original file.")
        # Replace the original file with the fixed one
        replace_successful = replace_file(temp_output_file, input_file)
        if not replace_successful:
            print(f"Replacement failed. The fixed data remains in '{temp_output_file}'.")
            sys.exit(1)
    else:
        print("Fixing process failed. Original file remains unchanged.")
        # Clean up temporary file if it exists and fixing failed
        if os.path.exists(temp_output_file):
            try:
                os.remove(temp_output_file)
                print(f"Removed temporary file '{temp_output_file}'.")
            except OSError as e_rem:
                print(f"Warning: Could not remove temporary file '{temp_output_file}'. Error: {e_rem}", file=sys.stderr)
        sys.exit(1)

    print(f"\nOriginal file '{input_file}' has been updated with attempted fixes.")
    print("You can now run the processing script (e.g., process_jsonl.py) on the updated file.") 