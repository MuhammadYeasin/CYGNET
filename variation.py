import os
import ollama
from typing import List, Dict, Tuple
import re

class ABCScale:
    def __init__(self, filename: str):
        """Parse scale info from filename like 'A,0,ascending,aeolian_scale,2octave,crotchets'"""
        basename = os.path.splitext(filename)[0]
        parts = basename.split(',')
        if len(parts) == 6:
            self.root_note = parts[0]
            self.transposition = int(parts[1])
            self.direction = parts[2]
            self.scale_type = parts[3]
            self.octave_span = parts[4]
            self.rhythm = parts[5]
        else:
            raise ValueError(f"Invalid filename format: {filename}")
    
    def __str__(self):
        return f"{self.root_note},{self.transposition},{self.direction},{self.scale_type},{self.octave_span},{self.rhythm}"

def create_variation_prompt(scale: ABCScale, abc_content: str) -> str:
    """Create prompt for generating variation while preserving format"""
    return f'''Create a new variation of this musical scale following these exact rules:

    1. MAINTAIN THESE PROPERTIES:
    - Root note: {scale.root_note}
    - Scale type: {scale.scale_type}
    - Direction: {scale.direction}
    - Octave span: {scale.octave_span}
    - Rhythm type: {scale.rhythm}
    
    2. KEEP EXACT FORMAT:
    - Keep all headers (X:, T:, C:, L:, M:, I:, K:, V:)
    - Maintain line breaks
    - Keep voice structure
    
    Original ABC notation:
    {abc_content}

    Respond with ONLY the new ABC notation, maintaining the exact same format.'''

def generate_variations(input_path: str, output_dir: str, num_variations: int = 3):
    """Generate variations for a single ABC file"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read original file
    with open(input_path, 'r') as f:
        original_content = f.read()
    
    # Parse scale information from filename
    filename = os.path.basename(input_path)
    scale = ABCScale(filename)
    
    # Save original file to output directory
    original_output = os.path.join(output_dir, f"{str(scale)}_original.abc")
    with open(original_output, 'w') as f:
        f.write(original_content)
    
    # Generate variations
    for i in range(num_variations):
        try:
            # Create prompt
            prompt = create_variation_prompt(scale, original_content)
            
            # Get response from Ollama
            response = ollama.chat('llama3.2', messages=[{
                'role': 'user',
                'content': prompt
            }])
            
            # Verify response contains required ABC elements
            abc_response = response.message.content.strip()
            required_elements = ['X:', 'T:', 'L:', 'M:', 'K:', 'V:']
            if all(element in abc_response for element in required_elements):
                # Save variation
                variation_file = os.path.join(output_dir, f"{str(scale)}_var{i+1}.abc")
                with open(variation_file, 'w') as f:
                    f.write(abc_response)
                print(f"Generated variation {i+1} for {filename}")
            else:
                print(f"Skipping invalid variation {i+1} for {filename} - missing required elements")
        
        except Exception as e:
            print(f"Error generating variation {i+1} for {filename}: {str(e)}")
            continue

def process_directory(input_dir: str, output_dir: str, num_variations: int = 3):
    """Process all ABC files in a directory"""
    # Verify input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Get all .abc files
    abc_files = [f for f in os.listdir(input_dir) if f.endswith('.abc')]
    
    if not abc_files:
        print(f"No .abc files found in {input_dir}")
        return
    
    print(f"Found {len(abc_files)} .abc files")
    
    for filename in abc_files:
        input_path = os.path.join(input_dir, filename)
        print(f"\nProcessing: {filename}")
        generate_variations(input_path, output_dir, num_variations)

def main():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configuration
    input_dir = input("Enter the path to your ABC files directory: ").strip()
    if not input_dir:
        input_dir = current_dir  # Use current directory if no input
    
    output_dir = os.path.join(current_dir, 'generated_scales')
    num_variations = 3
    
    try:
        process_directory(input_dir, output_dir, num_variations)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()