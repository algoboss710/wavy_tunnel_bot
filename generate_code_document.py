import os

def generate_code_document(root_dir, excluded_folders=None, excluded_files=None):
    if excluded_folders is None:
        excluded_folders = []
    if excluded_files is None:
        excluded_files = []

    document = "<documents>\n"
    index = 1

    for root, dirs, files in os.walk(root_dir):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_folders]
        for file in files:
            if file.endswith(".py") and file not in excluded_files:
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    file_content = f.read()

                document += f"<document index=\"{index}\">\n"
                document += f"<source>{file_path}</source>\n"
                document += "<document_content>\n"
                document += file_content
                document += "\n</document_content>\n"
                document += "</document>\n\n"
                index += 1

    document += "</documents>"
    return document

if __name__ == "__main__":
    # Specify the root directory of your repository
    root_directory = r"C:\Users\Owner\Desktop\upwork_projects\wavy_tunnel_bot"

    # Lists of folders and files to exclude
    excluded_folders = ['ignore_extra_stuff', 'jupyter_notebooks', '__pycache__']
    excluded_files = ['tempCodeRunnerFile.py', 'app.log','*.git','.gitignore','code_document.xml','generate_code_document.py','historical_data_cache.sqlite']

    # Generate the code document
    code_document = generate_code_document(root_directory, excluded_folders, excluded_files)

    # Specify the output file path
    output_file = os.path.join(root_directory, "code_document.xml")

    # Save the code document to a file
    with open(output_file, "w") as file:
        file.write(code_document)

    print(f"Code document saved to: {output_file}")
