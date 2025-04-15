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
    
    # Define the patterns to find and replace
    to_find = 'try:\n    import matplotlib.pyplot as plt\nexcept (ModuleNotFoundError, ImportError):'
    to_replace = 'try:\n    import matplotlib.pyplot as plt\n    import seaborn as sns\n    # Register seaborn styles\n    sns.set_theme()\n    try:\n        plt.style.use("seaborn-deep")\n    except OSError:\n        plt.style.use("default")\nexcept (ModuleNotFoundError, ImportError):'
    
    # Alternative pattern if first one not found
    alt_find = 'import matplotlib.pyplot as plt'
    alt_replace = 'import matplotlib.pyplot as plt\nimport seaborn as sns\n# Register seaborn styles\nsns.set_theme()\ntry:\n    plt.style.use("seaborn-deep")\nexcept OSError:\n    plt.style.use("default")'
    
    patched_content = None
    if to_find in content:
        patched_content = content.replace(to_find, to_replace)
    elif alt_find in content:
        patched_content = content.replace(alt_find, alt_replace)
    else:
        print("Could not find expected import pattern in plotting.py")
        return False
    
    # Backup the original file
    backup_path = plotting_path + '.bak'
    try:
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"Backed up original file to: {backup_path}")
    except Exception as e:
        print(f"Warning: Could not create backup: {e}")
    
    # Write the patched content
    with open(plotting_path, 'w') as f:
        f.write(patched_content)
    
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
