import os
import subprocess

from langchain_core.tools import tool

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
    """Only use this tool when the user explicitly commands you to generate a new local file on their machine 
    (e.g., 'Write this code to a python file' or 'Create a text document'). Do not use this tool to save internal notes 
    or conversational context.
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

@tool
def execute_terminal(command: str) -> str:
    """
    Executes a terminal/shell command and returns the standard output or error.
    Use this strictly to run Python scripts, git commands, pip installs, or check system status.
    """
    # 1. The Visual Warning
    print("\n" + "⚠️ "*20)
    print("🚨 EDITH IS REQUESTING TERMINAL ACCESS 🚨")
    print(f"👉 Command: {command}")
    print("⚠️ "*20)
    
    # 2. The Manual Code Block
    # This completely pauses the Python thread until you type on your physical keyboard.
    permission = input("Boss, do you authorize this command? (y/n): ").strip().lower()
    
    if permission != 'y' and permission != 'yes':
        print("🚫 Command blocked by Boss.")
        # 3. The Feedback Loop
        # We MUST return a string back to LangGraph. If we just exit, the AI crashes.
        # Returning this string tells the AI it was denied, preventing it from hallucinating that it succeeded.
        return f"Action Denied: The user refused to give permission to run the command '{command}'."
        
    print("✅ Executing command...")
    
    # 4. The Execution (Only runs if permission was 'y')
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            text=True, 
            capture_output=True, 
            timeout=15 
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            return f"Success:\n{output}" if output else "Success: Command executed with no output."
        
        else:
            return f"Command Failed (Code {result.returncode}):\n{result.stderr.strip()}"
            
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 15 seconds."
    except Exception as e:
        return f"System Error executing command: {str(e)}"