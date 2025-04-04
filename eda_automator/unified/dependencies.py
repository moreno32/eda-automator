"""
Module for checking dependencies required for report generation.
"""

import sys
import importlib.util
import shutil

def check_dependencies():
    """
    Check for required dependencies for different report formats.
    
    Returns:
        dict: Dictionary with boolean flags for each dependency
    """
    dependencies = {
        'imgkit': False,
        'selenium': False,
        'weasyprint': False,
        'pillow': False,
        'wkhtmltopdf': False,
    }
    
    # Check Python packages
    try:
        import imgkit
        dependencies['imgkit'] = True
    except ImportError:
        pass
    
    try:
        import selenium
        dependencies['selenium'] = True
    except ImportError:
        pass
    
    try:
        import weasyprint
        dependencies['weasyprint'] = True
    except ImportError:
        pass
    
    try:
        import PIL
        dependencies['pillow'] = True
    except ImportError:
        pass
    
    # Check for wkhtmltopdf executable
    if shutil.which('wkhtmltopdf'):
        dependencies['wkhtmltopdf'] = True
    
    return dependencies

def print_dependencies_status():
    """
    Print the status of all required dependencies.
    """
    dependencies = check_dependencies()
    
    print("\nDependencies Status:")
    print("--------------------")
    
    # Format for display
    status_format = "{:<15} {:<10}"
    print(status_format.format("Dependency", "Status"))
    print("-" * 25)
    
    for dep, installed in dependencies.items():
        status = "✓ Installed" if installed else "✗ Missing"
        print(status_format.format(dep, status))
    
    # Print recommendations
    print("\nRecommendations:")
    if not dependencies['imgkit'] or not dependencies['wkhtmltopdf']:
        print("- For image reports: pip install imgkit and install wkhtmltopdf")
    
    if not dependencies['selenium']:
        print("- Alternative for images: pip install selenium webdriver-manager")
    
    if not dependencies['weasyprint']:
        print("- Alternative for images: pip install weasyprint pdf2image")
    
    if not dependencies['pillow']:
        print("- For image processing: pip install pillow")
    
    print("\n")

if __name__ == "__main__":
    print_dependencies_status() 