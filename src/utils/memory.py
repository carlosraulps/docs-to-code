import json
import os
import hashlib
from typing import Dict, Any, Optional

class Memory:
    def __init__(self, log_path: Optional[str] = None):
        if not log_path:
            cache_dir = os.path.expanduser("~/.cache/docs-to-code")
            os.makedirs(cache_dir, exist_ok=True)
            self.log_path = os.path.join(cache_dir, "mcp_memory.json")
        else:
            self.log_path = log_path
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Loads the processed log from disk."""
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Warning: corrupted log file. Starting fresh.")
                return {}
        return {}

    def save_state(self):
        """Persists the current state to disk."""
        with open(self.log_path, 'w') as f:
            json.dump(self.state, f, indent=2)

    def is_processed(self, image_path: str) -> bool:
        """
        Checks if an image has already been processed.
        Uses a hash of the file path (or content hash for robustness) as key.
        For simplicity and speed, we check if the filename exists in the registry 
        AND if the mtime matches (to detect updates).
        """
        file_id = self._get_file_id(image_path)
        if file_id in self.state:
            # Check for "Poisoned State": existing error message
            content = self.state[file_id].get('content', "")
            
            # Handle both string (old cache) and dict (new cache) formats
            error_found = False
            if isinstance(content, str):
                if "% Error processing image" in content:
                    error_found = True
            elif isinstance(content, dict):
                # Check if "latex" or "markdown" values contain the error
                if "% Error processing image" in content.get("latex", "") or "% Error processing image" in content.get("markdown", ""):
                    error_found = True
                    
            if error_found:
                print(f"Retrying failed image: {os.path.basename(image_path)}")
                return False

            # Check if file has been modified since last process
            last_mtime = self.state[file_id].get('mtime', 0)
            current_mtime = os.path.getmtime(image_path)
            if current_mtime <= last_mtime:
                return True
        return False

    def get_cached_content(self, image_path: str) -> str:
        """Retrieves cached content for a processed image."""
        file_id = self._get_file_id(image_path)
        return self.state.get(file_id, {}).get('content', "")

    def mark_processed(self, image_path: str, content: str):
        """Updates the registry with the processed image data."""
        file_id = self._get_file_id(image_path)
        self.state[file_id] = {
            'file_path': image_path,
            'mtime': os.path.getmtime(image_path),
            'content': content
        }
        self.save_state()

    def _get_file_id(self, file_path: str) -> str:
        """Generates a unique ID for the file based on its absolute path."""
        return hashlib.sha256(os.path.abspath(file_path).encode()).hexdigest()
