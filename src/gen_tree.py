import os
from pathlib import Path

# Automatically ignore system, environment, and cache files
IGNORE_PATTERNS = {
    ".git", ".venv", "__pycache__", ".DS_Store", 
    ".pytest_cache", "gen_tree.py", "raw_data", "dataset_split", ".secret", "outputs"
}

def build_tree(dir_path: Path, prefix: str = "") -> str:
    tree_str = ""
    # Fetch and sort items (directories first, then files alphabetically)
    items = [p for p in dir_path.iterdir() if p.name not in IGNORE_PATTERNS]
    items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
    
    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        name = f"{item.name}/" if item.is_dir() else item.name
        tree_str += f"{prefix}{connector}{name}\n"
        
        if item.is_dir():
            extension = "    " if is_last else "│   "
            tree_str += build_tree(item, prefix + extension)
            
    return tree_str

if __name__ == "__main__":
    root_dir = Path('.')
    print(f"{root_dir.resolve().name}/")
    print(build_tree(root_dir), end="")