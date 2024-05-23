import os

def generate_code_document(root_dir):
    document = "<documents>\n"
    index = 1

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
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
    root_directory = r"C:\Users\16198\Desktop\automation\upwork\wavy\wavy_tunnel_bot"

    # Generate the code document
    code_document = generate_code_document(root_directory)

    # Specify the output file path
    output_file = os.path.join(root_directory, "code_document.xml")

    # Save the code document to a file
    with open(output_file, "w") as file:
        file.write(code_document)

    print(f"Code document saved to: {output_file}")
