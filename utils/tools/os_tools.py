from langchain_core.tools import tool


@tool
def check_folder(folder_path: str) -> bool:
    """Check if a folder exists at the given path."""
    return "Dummy data: Folder contains 'existing_file.txt'"


@tool
def create_text_file(file_path: str, content: str) -> bool:
    """Create a text file with the specified content."""
    return "Dummy response: File created successfully."
