"""Script to update import paths in all notebooks after reorganization."""

import json
from pathlib import Path

def update_notebook_imports(notebook_path):
    """Update imports in a single notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Check if this cell has llm_services import
            if 'from llm_services import' in source:
                # Update the source
                lines = cell['source']
                new_lines = []
                
                # Flag to track if we've added sys.path
                added_sys_path = False
                
                for i, line in enumerate(lines):
                    # Add sys.path before from llm_services
                    if 'from llm_services import' in line and not added_sys_path:
                        # Insert sys.path manipulation before this line
                        # Find where to insert (after imports like Path, load_dotenv)
                        insert_idx = i
                        
                        # Back up to find the right place (after other imports)
                        new_lines.append("import sys\n")
                        new_lines.append("from pathlib import Path\n")
                        new_lines.append("from dotenv import load_dotenv\n")
                        new_lines.append("\n")
                        new_lines.append("# Add parent directory to path to import from src\n")
                        new_lines.append("sys.path.append(str(Path.cwd().parent))\n")
                        new_lines.append("\n")
                        added_sys_path = True
                        
                        # Skip original import lines we just added
                        continue
                    
                    # Skip import sys, Path, load_dotenv if already present
                    if line.strip().startswith(('import sys', 'from pathlib import Path', 'from dotenv import load_dotenv')):
                        continue
                    
                    # Update llm_services import
                    if 'from llm_services import' in line:
                        line = line.replace('from llm_services import', 'from src.services.llm_services import')
                    
                    # Update config path
                    if 'load_config("config.yaml")' in line:
                        line = line.replace('load_config("config.yaml")', 'load_config("../src/config/config.yaml")')
                    
                    new_lines.append(line)
                
                if new_lines != lines:
                    cell['source'] = new_lines
                    modified = True
    
    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        return True
    return False

def main():
    """Update all notebooks."""
    notebooks_dir = Path(__file__).parent.parent.parent / "notebooks"
    
    notebooks = list(notebooks_dir.glob("*.ipynb"))
    
    print(f"Found {len(notebooks)} notebooks to update...")
    
    for nb_path in sorted(notebooks):
        print(f"Processing: {nb_path.name}")
        if update_notebook_imports(nb_path):
            print(f"  ✅ Updated")
        else:
            print(f"  ℹ️  No changes needed")
    
    print("\n✅ All notebooks updated!")

if __name__ == "__main__":
    main()

