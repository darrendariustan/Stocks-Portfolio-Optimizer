"""
Patch script for PyPortfolioOpt to fix seaborn-deep style issue.
This script modifies the plotting.py file to safely handle the seaborn-deep style.
Run this script after installing the PyPortfolioOpt package.
"""

import os
import sys
import site
import glob

def find_plotting_module():
    """Find the location of the PyPortfolioOpt plotting.py file."""
    # Check for common install locations
    potential_paths = []
    
    # Virtual env paths (Render uses a virtual env)
    if 'VIRTUAL_ENV' in os.environ:
        venv_path = os.path.join(
            os.environ.get('VIRTUAL_ENV'),
            'lib',
            f'python{sys.version_info.major}.{sys.version_info.minor}',
            'site-packages',
            'pypfopt',
            'plotting.py'
        )
        potential_paths.append(venv_path)
    
    # Site-packages paths
    for site_dir in site.getsitepackages():
        potential_paths.append(os.path.join(site_dir, 'pypfopt', 'plotting.py'))
    
    # User site-packages
    if site.USER_SITE:
        potential_paths.append(os.path.join(site.USER_SITE, 'pypfopt', 'plotting.py'))
    
    # Search for all pypfopt installations using glob
    for site_dir in site.getsitepackages():
        for path in glob.glob(os.path.join(site_dir, '**', 'pypfopt', 'plotting.py'), recursive=True):
            potential_paths.append(path)
    
    # Check if paths exist
    for path in potential_paths:
        if os.path.exists(path):
            print(f"Found PyPortfolioOpt plotting module at: {path}")
            return path
    
    print("Could not find PyPortfolioOpt plotting module. Is it installed?")
    return None

def patch_pypfopt_plotting(plotting_path):
    """Patch the pyplot style use in PyPortfolioOpt plotting module."""
    if not plotting_path or not os.path.exists(plotting_path):
        return False
    
    # Read the file
    with open(plotting_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'except OSError:' in content and 'plt.style.use("default")' in content:
        print("PyPortfolioOpt plotting module already patched.")
        return True
    
    # Create a backup of the file
    backup_path = plotting_path + '.bak'
    try:
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"Backed up original file to: {backup_path}")
    except Exception as e:
        print(f"Warning: Could not create backup: {e}")
    
    # Most drastic approach: replace the import of matplotlib.pyplot with a safe version
    # This avoids any issues with splitting docstrings or code structure
    new_content = content.replace(
        'import matplotlib.pyplot as plt',
        '''import matplotlib.pyplot as plt
import seaborn as sns
# Register seaborn styles
sns.set_theme()
try:
    plt.style.use("seaborn-deep")
except OSError:
    plt.style.use("default")'''
    )
    
    # Write the patched content
    with open(plotting_path, 'w') as f:
        f.write(new_content)
    
    print("Successfully patched PyPortfolioOpt plotting module!")
    return True

if __name__ == "__main__":
    plotting_path = find_plotting_module()
    if plotting_path:
        success = patch_pypfopt_plotting(plotting_path)
        if success:
            print("✅ Patch completed successfully!")
        else:
            print("❌ Patch failed.")
            sys.exit(1)
    else:
        print("❌ Could not find PyPortfolioOpt module to patch.")
        sys.exit(1)
    
    sys.exit(0)
