import os

def generate_specific_files_document(root_dir, target_files, ignore_dirs=None):
    if ignore_dirs is None:
        ignore_dirs = ['venv', '.venv', '__pycache__', '.git', '.mypy_cache', '.pytest_cache']

    document = ""

    for root, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to exclude ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for file in files:
            if file in target_files:
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    file_content = f.read()

                document += f"File: {file_path}\n"
                document += "----------------------------------------\n"
                document += file_content
                document += "\n\n"

    return document

if __name__ == "__main__":
    # Specify the root directory of your repository
    root_directory = r"C:\Users\owner\Desktop\upwork_projects\trade_bot\wavy_tunnel_bot"

    # List of specific files to include (with their exact filenames)
    target_files = ['main.py', 'config.py', 'tunnel_strategy.py', 'backtest.py', 'data_retrieval.py' '.env']

    # Generate the document for the specified files, ignoring virtual env and other unnecessary directories
    specific_files_document = generate_specific_files_document(root_directory, target_files)

    # Specify the output file path
    output_file = os.path.join(root_directory, "specific_files_document.txt")

    # Save the document to a file
    with open(output_file, "w") as file:
        file.write(specific_files_document)

    print(f"Specific files document saved to: {output_file}")
