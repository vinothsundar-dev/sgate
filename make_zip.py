"""Utility function to create a zip archive of the project."""

import os
import zipfile
from pathlib import Path
from typing import Optional, List


def make_zip(
    source_dir: str = ".",
    output_name: str = "sgate.zip",
    exclude_patterns: Optional[List[str]] = None,
    verbose: bool = True
) -> str:
    """
    Create a zip archive of the project directory.
    
    Args:
        source_dir: Source directory to zip (default: current directory)
        output_name: Name of the output zip file (default: "sgate.zip")
        exclude_patterns: List of patterns to exclude (e.g., ['.git', '__pycache__', '*.pyc'])
        verbose: Print progress information (default: True)
    
    Returns:
        Path to the created zip file
    
    Example:
        >>> zip_path = make_zip()
        >>> print(f"Created: {zip_path}")
        
        >>> zip_path = make_zip(exclude_patterns=['.git', '.venv', '*.pyc', '__pycache__'])
    """
    if exclude_patterns is None:
        exclude_patterns = ['.git', '.venv', '__pycache__', '*.pyc', '.DS_Store', '*.egg-info']
    
    source_path = Path(source_dir).resolve()
    output_path = source_path / output_name
    
    if verbose:
        print(f"Creating zip: {output_path}")
    
    def should_exclude(file_path: Path) -> bool:
        """Check if file matches any exclude pattern."""
        path_str = str(file_path)
        name = file_path.name
        for pattern in exclude_patterns:
            if pattern.startswith('*'):
                if name.endswith(pattern[1:]):
                    return True
            elif pattern in path_str:
                return True
        return False
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not should_exclude(Path(root) / d)]
            
            for file in files:
                file_path = Path(root) / file
                if not should_exclude(file_path):
                    arcname = file_path.relative_to(source_path)
                    zf.write(file_path, arcname=arcname)
                    if verbose:
                        print(f"  added: {arcname}")
    
    zip_size_mb = output_path.stat().st_size / (1024 * 1024)
    if verbose:
        print(f"✓ Created {output_path.name} ({zip_size_mb:.2f} MB)")
    
    return str(output_path)


if __name__ == "__main__":
    # Example usage
    zip_file = make_zip(exclude_patterns=[
        '.git', '.venv', '__pycache__', '*.pyc', 
        '.DS_Store', '*.egg-info', '.pytest_cache'
    ])
    print(f"\nZip archive created: {zip_file}")
