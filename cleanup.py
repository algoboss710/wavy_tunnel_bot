import re

# Read the script from a file
with open("output_script.py", "r") as file:
    script = file.read()

# Remove extra line spacing
script_cleaned = re.sub(r'\n\s*\n', '\n', script)

# Write the cleaned script back to the file
with open("output_script.py", "w") as file:
    file.write(script_cleaned)
