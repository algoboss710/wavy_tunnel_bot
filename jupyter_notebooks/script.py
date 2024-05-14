import os
import json

def extract_code_and_markdown(ipynb_file, output_file):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ipynb_path = os.path.join(script_dir, ipynb_file)
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                # Write markdown content as comments
                f.write("# " + ''.join(cell['source']) + '\n\n')
            elif cell['cell_type'] == 'code':
                # Write code cells
                f.write('\n'.join(cell['source']) + '\n\n')

if __name__ == "__main__":
    ipynb_file = "full_strategy_conversion.ipynb"
    output_file = "output_script.py"
    extract_code_and_markdown(ipynb_file, output_file)
