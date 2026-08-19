from pathlib import Path

def print_tree(directory: Path, prefix: str = ""):
    # Get items in directory, sorted with directories first
    items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    
    # Process each item
    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        print(f"{prefix}{connector}{item.name}")
        
        # Recursively display contents if item is a directory
        if item.is_dir():
            extension = "    " if is_last else "│   "
            print_tree(item, prefix + extension)

if __name__ == "__main__":
    # Display tree starting from the current directory (.)
    current_dir = Path(".")
    print(f"{current_dir.resolve().name}/")
    print_tree(current_dir)