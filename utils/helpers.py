import yaml
import os

def read_yaml_config(file_path):
    """
    Reads a YAML file and returns the content as a Python dictionary.
    Returns None if the file is missing or contains invalid syntax.
    """
    if not os.path.exists(file_path):
        print(f"Error: Config file not found at '{file_path}'")
        return None

    try:
        with open(file_path, 'r') as file:
            # safe_load converts YAML to Python dicts/lists
            config = yaml.safe_load(file)
            return config
            
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# --- Example Usage ---
# data = read_yaml_config("config.yaml")
# if data:
#     print(data['voice_settings']['model'])