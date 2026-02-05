from pathlib import Path

def print_tree(directory: Path, prefix: str = ""):
    # ディレクトリ内のアイテムを取得してソート（ディレクトリ優先）
    items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    
    # 各アイテムについて処理
    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        print(f"{prefix}{connector}{item.name}")
        
        # ディレクトリなら再帰的に中身を表示
        if item.is_dir():
            extension = "    " if is_last else "│   "
            print_tree(item, prefix + extension)

if __name__ == "__main__":
    # カレントディレクトリ (.) を起点に表示
    current_dir = Path(".")
    print(f"{current_dir.resolve().name}/")
    print_tree(current_dir)