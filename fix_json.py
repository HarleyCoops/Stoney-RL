import json
import re

# Function to properly escape quotes in JSON strings
def fix_jsonl_file(input_file, output_file):
    print(f"Processing {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        line_count = 0
        fixed_count = 0
        
        for line in infile:
            line_count += 1
            if line_count % 10000 == 0:
                print(f"Processed {line_count} lines...")
            
            # Skip empty lines
            if not line.strip():
                continue
                
            try:
                # Try to parse the JSON to check if it's valid
                json.loads(line)
                # If parsing succeeds, write the line as is
                outfile.write(line)
            except json.JSONDecodeError:
                # Fix unescaped quotes in the line
                fixed_line = line
                
                # Find all field contents (everything between field quotes)
                pattern = r'":\s*"(.*?)"(?=,|\s*})'
                
                def fix_quotes(match):
                    content = match.group(1)
                    # Replace unescaped quotes that aren't already escaped
                    # This regex looks for quotes not preceded by a backslash
                    fixed_content = re.sub(r'(?<!\\)"', r'\\"', content)
                    return '": "' + fixed_content + '"'
                
                fixed_line = re.sub(pattern, fix_quotes, fixed_line)
                
                # Check if the fix worked
                try:
                    json.loads(fixed_line)
                    outfile.write(fixed_line)
                    fixed_count += 1
                except json.JSONDecodeError as e:
                    print(f"Failed to fix line {line_count}: {str(e)}")
                    # Write the original line with a comment
                    outfile.write(f"// ERROR LINE {line_count}: {line}")
        
        print(f"Completed! Fixed {fixed_count} of {line_count} lines.")

if __name__ == "__main__":
    input_file = "synthetic_stoney_data.jsonl"
    output_file = "synthetic_stoney_data_fixed.jsonl"
    fix_jsonl_file(input_file, output_file) 