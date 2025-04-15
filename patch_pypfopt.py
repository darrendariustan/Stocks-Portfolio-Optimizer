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
    
    # Import matplotlib section
    if 'import matplotlib.pyplot as plt' in content:
        # Read the file line by line to ensure proper patching
        with open(plotting_path, 'r') as f:
            lines = f.readlines()
        
        # Find the matplotlib import line
        for i, line in enumerate(lines):
            if 'import matplotlib.pyplot as plt' in line and 'try:' not in line:
                # Simple case - just an import without try/except
                lines[i] = 'import matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Register seaborn styles\nsns.set_theme()\ntry:\n    plt.style.use("seaborn-deep")\nexcept OSError:\n    plt.style.use("default")\n'
                break
            elif 'import matplotlib.pyplot as plt' in line and i > 0 and 'try:' in lines[i-1]:
                # Complex case - import is inside a try/except block
                # Add seaborn after matplotlib but before the except
                lines[i] = '    import matplotlib.pyplot as plt\n    import seaborn as sns\n    # Register seaborn styles\n    sns.set_theme()\n    try:\n        plt.style.use("seaborn-deep")\n    except OSError:\n        plt.style.use("default")\n'
                break
        
        # Write the modified content back
        with open(plotting_path, 'w') as f:
            f.writelines(lines)
        
        print("Successfully patched PyPortfolioOpt plotting module!")
        return True
    else:
        print("Could not find matplotlib import in plotting.py")
        return False

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
