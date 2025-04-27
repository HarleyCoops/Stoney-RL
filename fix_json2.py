import json
import re

def fix_jsonl_file(input_file, output_file):
    print(f"Processing {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        line_count = 0
        fixed_count = 0
        error_count = 0
        
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
                try:
                    # Method 1: Fix unescaped quotes in string values
                    fixed_line = line
                    
                    # Extract all potential field values
                    field_pattern = r'":\s*"(.*?)"(?=,|\s*})'
                    
                    def fix_quotes(match):
                        content = match.group(1)
                        # Replace unescaped quotes that aren't already escaped
                        fixed_content = re.sub(r'(?<!\\)"', r'\\"', content)
                        return '": "' + fixed_content + '"'
                    
                    fixed_line = re.sub(field_pattern, fix_quotes, fixed_line)
                    
                    # Try if the fix worked
                    try:
                        json.loads(fixed_line)
                        outfile.write(fixed_line)
                        fixed_count += 1
                        continue
                    except json.JSONDecodeError:
                        pass  # If this method failed, try the next method
                    
                    # Method 2: More aggressive fix - manual structure reconstruction
                    try:
                        # Extract the components we can identify with high confidence
                        question_match = re.search(r'"question":\s*"(.*?)(?<!\\)"(?=,)', line)
                        answer_match = re.search(r'"answer":\s*"(.*?)(?<!\\)"(?=,)', line)
                        generated_at_match = re.search(r'"generated_at":\s*"(.*?)(?<!\\)"(?=,)', line)
                        method_match = re.search(r'"method":\s*"(.*?)(?<!\\)"(?=,|\s*})', line)
                        original_id_match = re.search(r'"original_id":\s*(\d+)(?=\s*})', line)
                        
                        # If we found most components, reconstruct the JSON
                        if question_match and answer_match and (method_match or original_id_match):
                            question = question_match.group(1).replace('"', '\\"')
                            answer = answer_match.group(1).replace('"', '\\"')
                            
                            # Default values if not found
                            generated_at = generated_at_match.group(1) if generated_at_match else "unknown"
                            method = method_match.group(1) if method_match else "unknown"
                            original_id = original_id_match.group(1) if original_id_match else "0"
                            
                            # Construct a valid JSON object
                            reconstructed = '{' + f'"question": "{question}", "answer": "{answer}", "generated_at": "{generated_at}", "method": "{method}", "original_id": {original_id}' + '}\n'
                            
                            # Verify it's valid JSON
                            json.loads(reconstructed)
                            outfile.write(reconstructed)
                            fixed_count += 1
                            continue
                    except (json.JSONDecodeError, AttributeError, IndexError):
                        pass  # If reconstruction failed, fall back to the next method
                    
                    # Method 3: Last resort - use regex to fix JSON syntax issues
                    try:
                        # Fix missing quotes around property names
                        property_fix = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', fixed_line)
                        
                        # Fix trailing commas
                        comma_fix = re.sub(r',\s*}', '}', property_fix)
                        
                        # Try if the fixes worked
                        json.loads(comma_fix)
                        outfile.write(comma_fix)
                        fixed_count += 1
                        continue
                    except json.JSONDecodeError:
                        pass
                    
                    # If we got here, none of the methods worked
                    outfile.write(f"// ERROR LINE {line_count}: {line}")
                    error_count += 1
                
                except Exception as e:
                    print(f"Failed to fix line {line_count}: {str(e)}")
                    outfile.write(f"// ERROR LINE {line_count}: {line}")
                    error_count += 1
        
        print(f"Completed! Fixed {fixed_count} of {line_count} lines. Errors: {error_count}")

if __name__ == "__main__":
    input_file = "synthetic_stoney_data.jsonl"
    output_file = "synthetic_stoney_data_fixed2.jsonl"
    fix_jsonl_file(input_file, output_file) 