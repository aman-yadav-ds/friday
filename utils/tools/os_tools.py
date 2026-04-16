from langchain_core.tools import tool
import os

@tool
def check_folder(folder_path: str) -> bool:
    """
    Checks if a folder exists and lists its files. 
    Always use this before creating files to see what is already there
    """

    folder_path = os.path.expanduser(folder_path)  # Expand ~ to the user home directory
    print(f"Folder path after expansion: {os.path.abspath(folder_path)}")  # Debug print to check the path
    # 1. Check if it exists at all
    if not os.path.exists(folder_path):
        return f"The folder '{folder_path}' does not exist."
    
    # 2. Check if it's actually a folder
    if not os.path.isdir(folder_path):
        return f"The path '{folder_path}' exists but is not a folder."
    
    # 3. List the files in the folder
    try:
        files = os.listdir(folder_path)
        if not files:
            return f"The folder '{folder_path}' exists but is empty."
        return f"The folder '{folder_path}' exists and contains the following files: {', '.join(files)}"
    except Exception as e:
        return f"Error accessing the folder '{folder_path}': {str(e)}"


@tool
def create_file(file_path: str, content: str) -> bool:
    """
    Creates a new file with the given content at the specified path.
    """
    file_path = os.path.expanduser(file_path)  # Expand ~ to the user home directory

    # 1. Safety Check: Don't overwrite existing files blindly
    if os.path.exists(file_path):
        return f"Warning: A file already exists at '{file_path}'. Do not overwrite it. Ask the user what to do."
    
    try:
        # 2. Create the folder if it doesn't exist yet
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(content)
        return f"Success! The file was created exactly at '{file_path}'."
    except Exception as e:
        return f"Error creating the file '{file_path}': {str(e)}"