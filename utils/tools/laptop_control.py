import sys
import os
import subprocess
from langchain.tools import tool

@tool
def open_app(self, app_name: str) -> str:
    """
    Open a laptop application by name.
    """
    try:
        if sys.platform == "win32":
            os.startfile(app_name)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-a", app_name])
        elif sys.platform == "linux":
            subprocess.Popen([app_name])
        else:
            return f"Error: Unsupported OS '{sys.platform}'"
        return f"Opened application: {app_name}"
    except Exception as e:
        return f"An error occurred while trying to open the application: {str(e)}"
    
@tool
def close_app(self, app_name: str) -> str:
    """
    Close a laptop application by name.
    """
    try:
        if sys.platform == "win32":
            subprocess.Popen(["taskkill", "/IM", f"{app_name}.exe", "/F"])
        elif sys.platform == "darwin" or sys.platform == "linux":
            subprocess.Popen(["pkill", app_name])
        else:
            return f"Error: Unsupported OS '{sys.platform}'"
        return f"Closed application: {app_name}"
    except Exception as e:
        return f"An error occurred while trying to close the application: {str(e)}"