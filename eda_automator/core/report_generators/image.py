"""
Image report generation module for EDA Automator

This module provides functions for generating image reports from HTML content.
"""

import os
import sys
import tempfile
import importlib
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from eda_automator.core.utils.logger import get_logger
from eda_automator.core.utils.dependencies import check_module, check_external_tool

# Initialize logger
logger = get_logger()

# Define dependency requirements
REQUIREMENTS = {
    'imgkit': {
        'module': 'imgkit',
        'min_version': '1.2.2',
        'install_command': 'pip install imgkit>=1.2.2',
        'import_error': "imgkit is required for image generation using wkhtmltopdf"
    },
    'selenium': {
        'module': 'selenium',
        'min_version': '4.1.0',
        'install_command': 'pip install selenium>=4.1.0 webdriver-manager>=3.5.2',
        'import_error': "selenium is required for image generation using browser rendering"
    },
    'weasyprint': {
        'module': 'weasyprint',
        'min_version': '53.0',
        'install_command': 'pip install weasyprint>=53.0',
        'import_error': "weasyprint is required for high-quality image generation"
    },
    'pdf2image': {
        'module': 'pdf2image',
        'min_version': '1.16.0',
        'install_command': 'pip install pdf2image>=1.16.0 pillow>=8.3.1',
        'import_error': "pdf2image is required for converting PDF to images"
    }
}

# Define external tool requirements
EXTERNAL_TOOLS = {
    'wkhtmltopdf': {
        'command': 'wkhtmltopdf --version',
        'install_guide': 'Please install wkhtmltopdf: https://wkhtmltopdf.org/downloads.html',
        'error_message': "wkhtmltopdf is required for image generation using imgkit"
    }
}

def check_dependencies(method=None):
    """
    Check if the required dependencies are installed based on the specified method.
    
    Parameters
    ----------
    method : str, optional
        The method to use for image generation ('imgkit', 'selenium', 'weasyprint').
        If None, check all dependencies.
        
    Returns
    -------
    dict
        Dictionary with status of each dependency
    """
    dependencies = {}
    
    # Check specific dependencies based on method
    if method == 'imgkit' or method is None:
        dependencies['imgkit'] = check_module(
            REQUIREMENTS['imgkit']['module'],
            REQUIREMENTS['imgkit']['min_version']
        )
        dependencies['wkhtmltopdf'] = check_external_tool(
            EXTERNAL_TOOLS['wkhtmltopdf']['command']
        )
    
    if method == 'selenium' or method is None:
        dependencies['selenium'] = check_module(
            REQUIREMENTS['selenium']['module'],
            REQUIREMENTS['selenium']['min_version']
        )
    
    if method == 'weasyprint' or method is None:
        dependencies['weasyprint'] = check_module(
            REQUIREMENTS['weasyprint']['module'],
            REQUIREMENTS['weasyprint']['min_version']
        )
        dependencies['pdf2image'] = check_module(
            REQUIREMENTS['pdf2image']['module'],
            REQUIREMENTS['pdf2image']['min_version']
        )
    
    return dependencies

def get_missing_dependencies(method=None):
    """
    Get a list of missing dependencies for the specified method.
    
    Parameters
    ----------
    method : str, optional
        The method to use for image generation ('imgkit', 'selenium', 'weasyprint').
        If None, check all dependencies.
        
    Returns
    -------
    list
        List of missing dependencies
    """
    dependencies = check_dependencies(method)
    missing = []
    
    for name, available in dependencies.items():
        if not available:
            if name in REQUIREMENTS:
                missing.append({
                    'name': name,
                    'install_command': REQUIREMENTS[name]['install_command'],
                    'error_message': REQUIREMENTS[name]['import_error']
                })
            elif name in EXTERNAL_TOOLS:
                missing.append({
                    'name': name,
                    'install_guide': EXTERNAL_TOOLS[name]['install_guide'],
                    'error_message': EXTERNAL_TOOLS[name]['error_message']
                })
    
    return missing

def install_dependencies(method=None, stdout=True):
    """
    Attempt to install missing dependencies for the specified method.
    
    Parameters
    ----------
    method : str, optional
        The method to use for image generation ('imgkit', 'selenium', 'weasyprint').
        If None, install all dependencies.
    stdout : bool, default True
        Whether to print output to stdout
        
    Returns
    -------
    bool
        True if all dependencies were installed successfully, False otherwise
    """
    missing = get_missing_dependencies(method)
    
    if not missing:
        logger.info("All dependencies are already installed.")
        return True
    
    success = True
    
    for dep in missing:
        if 'install_command' in dep:
            try:
                logger.info(f"Installing {dep['name']}...")
                process = subprocess.run(
                    dep['install_command'],
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE if not stdout else None,
                    stderr=subprocess.PIPE if not stdout else None
                )
                logger.info(f"Successfully installed {dep['name']}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {dep['name']}: {e}")
                logger.error(f"Try manually running: {dep['install_command']}")
                success = False
        else:
            logger.warning(f"External tool {dep['name']} is required.")
            logger.warning(dep['install_guide'])
            success = False
    
    return success

def generate_image_report(
    output_path: str,
    html_content: Optional[str] = None,
    html_file: Optional[str] = None,
    data=None,
    results=None,
    figures=None,
    target_variable=None,
    settings=None,
    template_path=None,
    method='auto',
    orientation='landscape',
    width=None,
    height=None,
    quality=90,
    **kwargs
) -> str:
    """
    Generate an image report from HTML content.
    
    Parameters
    ----------
    output_path : str
        Path to save the image report
    html_content : str, optional
        HTML content to convert to image
    html_file : str, optional
        Path to HTML file to convert to image
    data : pandas.DataFrame, optional
        Data used for the analysis, used to generate HTML if no content/file provided
    results : dict, optional
        Results from the analysis, used to generate HTML if no content/file provided
    figures : dict, optional
        Visualization figures, used to generate HTML if no content/file provided
    target_variable : str, optional
        Name of the target variable, used to generate HTML if no content/file provided
    settings : dict, optional
        Settings for the report
    template_path : str, optional
        Path to a custom HTML template file, used to generate HTML if no content/file provided
    method : str, default 'auto'
        Method to use for image generation ('auto', 'imgkit', 'selenium', 'weasyprint')
    orientation : str, default 'landscape'
        Page orientation ('landscape' or 'portrait')
    width : int, optional
        Width of the image in pixels
    height : int, optional
        Height of the image in pixels
    quality : int, default 90
        Image quality (0-100)
        
    Returns
    -------
    str
        Path to the generated image
    """
    logger.info(f"Generating image report to {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Determine image dimensions
    if orientation == 'landscape':
        width = width or 1920
        height = height or 1080
    else:  # portrait
        width = width or 1080
        height = height or 1920
    
    # If no HTML content or file provided, generate HTML from data and results
    if not html_content and not html_file:
        # Import html module dynamically to avoid circular imports
        try:
            from eda_automator.core.report_generators.html import generate_html_report
            
            # Create temporary HTML file
            fd, html_file = tempfile.mkstemp(suffix='.html')
            os.close(fd)
            
            # Generate HTML report
            generate_html_report(
                output_path=html_file,
                data=data,
                results=results,
                figures=figures,
                target_variable=target_variable,
                settings=settings,
                template_path=template_path,
                **kwargs
            )
            
            # Set flag to clean up temporary file
            tmp_html = True
        except ImportError as e:
            logger.error(f"Failed to import HTML module: {e}")
            raise ImportError("HTML module is required for generating image reports from data and results")
    elif html_content:
        # Create temporary HTML file with provided content
        fd, html_file = tempfile.mkstemp(suffix='.html')
        os.close(fd)
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Set flag to clean up temporary file
        tmp_html = True
    else:
        # Using provided HTML file
        tmp_html = False
    
    # Choose appropriate method for image generation
    if method == 'auto':
        # Try each method in order of preference
        if check_module('weasyprint', REQUIREMENTS['weasyprint']['min_version']):
            method = 'weasyprint'
        elif check_module('imgkit', REQUIREMENTS['imgkit']['min_version']) and \
             check_external_tool(EXTERNAL_TOOLS['wkhtmltopdf']['command']):
            method = 'imgkit'
        elif check_module('selenium', REQUIREMENTS['selenium']['min_version']):
            method = 'selenium'
        else:
            # None of the preferred methods are available, attempt to install dependencies
            logger.warning("No suitable image generation method found. Attempting to install dependencies...")
            if install_dependencies(method='weasyprint'):
                method = 'weasyprint'
            elif install_dependencies(method='imgkit'):
                method = 'imgkit'
            elif install_dependencies(method='selenium'):
                method = 'selenium'
            else:
                raise ImportError(
                    "No suitable image generation method is available. "
                    "Please install one of the following:\n"
                    "- WeasyPrint: pip install weasyprint>=53.0 pdf2image>=1.16.0 pillow>=8.3.1\n"
                    "- imgkit: pip install imgkit>=1.2.2 and install wkhtmltopdf\n"
                    "- Selenium: pip install selenium>=4.1.0 webdriver-manager>=3.5.2"
                )
    
    # Generate image using the selected method
    try:
        if method == 'imgkit':
            return _generate_image_with_imgkit(
                html_file, output_path, orientation, width, height, quality
            )
        elif method == 'selenium':
            return _generate_image_with_selenium(
                html_file, output_path, orientation, width, height, quality
            )
        elif method == 'weasyprint':
            return _generate_image_with_weasyprint(
                html_file, output_path, orientation, width, height, quality
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
    finally:
        # Clean up temporary HTML file if needed
        if tmp_html and os.path.exists(html_file):
            try:
                os.remove(html_file)
            except Exception as e:
                logger.warning(f"Failed to remove temporary HTML file: {e}")

def _generate_image_with_imgkit(
    html_file, output_path, orientation, width, height, quality
):
    """
    Generate image using imgkit (wkhtmltopdf).
    """
    logger.info("Generating image using imgkit (wkhtmltopdf)")
    
    try:
        import imgkit
        
        # Build options for imgkit
        options = {
            'format': 'png',
            'encoding': 'UTF-8',
            'quiet': '',
        }
        
        # Set dimensions
        if orientation == 'landscape':
            options['orientation'] = 'Landscape'
        else:
            options['orientation'] = 'Portrait'
        
        if width and height:
            options['width'] = width
            options['height'] = height
        
        # Set quality
        if quality is not None:
            options['quality'] = quality
        
        # Generate image
        imgkit.from_file(html_file, output_path, options=options)
        
        logger.info(f"Generated image report at {output_path}")
        return output_path
    
    except ImportError as e:
        logger.error(f"Failed to import imgkit: {e}")
        logger.error(f"Install imgkit with: {REQUIREMENTS['imgkit']['install_command']}")
        logger.error(f"Install wkhtmltopdf from: https://wkhtmltopdf.org/downloads.html")
        raise

def _generate_image_with_selenium(
    html_file, output_path, orientation, width, height, quality
):
    """
    Generate image using Selenium WebDriver.
    """
    logger.info("Generating image using Selenium WebDriver")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        from PIL import Image
        
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f"--window-size={width},{height}")
        
        # Initialize WebDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        try:
            # Convert to file:// URL
            html_path = Path(html_file).resolve().as_uri()
            
            # Open the HTML file
            driver.get(html_path)
            
            # Wait for JavaScript to complete
            driver.implicitly_wait(2)
            
            # Get the rendered page dimensions
            page_width = driver.execute_script("return document.body.scrollWidth")
            page_height = driver.execute_script("return document.body.scrollHeight")
            
            # Set window size to accommodate the page
            driver.set_window_size(page_width, page_height)
            
            # Take screenshot
            driver.save_screenshot(output_path)
            
            # Optimize image if quality is specified
            if quality < 100:
                img = Image.open(output_path)
                img.save(output_path, optimize=True, quality=quality)
            
            logger.info(f"Generated image report at {output_path}")
            return output_path
        
        finally:
            # Close the driver
            driver.quit()
    
    except ImportError as e:
        logger.error(f"Failed to import selenium: {e}")
        logger.error(f"Install selenium with: {REQUIREMENTS['selenium']['install_command']}")
        raise

def _generate_image_with_weasyprint(
    html_file, output_path, orientation, width, height, quality
):
    """
    Generate image using WeasyPrint.
    """
    logger.info("Generating image using WeasyPrint")
    
    try:
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
        import pdf2image
        from PIL import Image
        
        # Configure fonts
        font_config = FontConfiguration()
        
        # Build CSS
        css_text = "@page { margin: 0; "
        
        if orientation == 'landscape':
            css_text += "size: landscape; "
        else:
            css_text += "size: portrait; "
        
        css_text += "}"
        
        css = CSS(string=css_text)
        
        # Generate PDF first
        pdf_file = output_path + '.pdf'
        HTML(filename=html_file).write_pdf(
            pdf_file,
            stylesheets=[css],
            font_config=font_config
        )
        
        # Convert PDF to image
        images = pdf2image.convert_from_path(
            pdf_file,
            dpi=300,
            output_folder=os.path.dirname(output_path),
            fmt='png',
            single_file=True
        )
        
        # Save the first page as the output image
        if images:
            # Resize if specific dimensions are required
            if width and height:
                images[0] = images[0].resize((width, height), Image.LANCZOS)
            
            # Set quality and save
            if quality < 100:
                images[0].save(output_path, optimize=True, quality=quality)
            else:
                images[0].save(output_path)
        
        # Clean up the temporary PDF file
        if os.path.exists(pdf_file):
            os.remove(pdf_file)
        
        logger.info(f"Generated image report at {output_path}")
        return output_path
    
    except ImportError as e:
        logger.error(f"Failed to import WeasyPrint or pdf2image: {e}")
        logger.error(f"Install WeasyPrint with: {REQUIREMENTS['weasyprint']['install_command']}")
        logger.error(f"Install pdf2image with: {REQUIREMENTS['pdf2image']['install_command']}")
        raise

def generate_landscape_report(
    output_path: str,
    html_content: Optional[str] = None,
    html_file: Optional[str] = None,
    data=None,
    results=None,
    figures=None,
    target_variable=None,
    settings=None,
    template_path=None,
    method='auto',
    width=1920,
    height=1080,
    quality=90,
    **kwargs
) -> str:
    """
    Generate a landscape-oriented image report.
    
    This is a convenience function that calls `generate_image_report` with orientation='landscape'.
    
    Parameters
    ----------
    output_path : str
        Path to save the image report
    html_content : str, optional
        HTML content to convert to image
    html_file : str, optional
        Path to HTML file to convert to image
    data : pandas.DataFrame, optional
        Data used for the analysis, used to generate HTML if no content/file provided
    results : dict, optional
        Results from the analysis, used to generate HTML if no content/file provided
    figures : dict, optional
        Visualization figures, used to generate HTML if no content/file provided
    target_variable : str, optional
        Name of the target variable, used to generate HTML if no content/file provided
    settings : dict, optional
        Settings for the report
    template_path : str, optional
        Path to a custom HTML template file, used to generate HTML if no content/file provided
    method : str, default 'auto'
        Method to use for image generation ('auto', 'imgkit', 'selenium', 'weasyprint')
    width : int, default 1920
        Width of the image in pixels
    height : int, default 1080
        Height of the image in pixels
    quality : int, default 90
        Image quality (0-100)
        
    Returns
    -------
    str
        Path to the generated image
    """
    return generate_image_report(
        output_path=output_path,
        html_content=html_content,
        html_file=html_file,
        data=data,
        results=results,
        figures=figures,
        target_variable=target_variable,
        settings=settings,
        template_path=template_path,
        method=method,
        orientation='landscape',
        width=width,
        height=height,
        quality=quality,
        **kwargs
    )

def generate_portrait_report(
    output_path: str,
    html_content: Optional[str] = None,
    html_file: Optional[str] = None,
    data=None,
    results=None,
    figures=None,
    target_variable=None,
    settings=None,
    template_path=None,
    method='auto',
    width=1080,
    height=1920,
    quality=90,
    **kwargs
) -> str:
    """
    Generate a portrait-oriented image report.
    
    This is a convenience function that calls `generate_image_report` with orientation='portrait'.
    
    Parameters
    ----------
    output_path : str
        Path to save the image report
    html_content : str, optional
        HTML content to convert to image
    html_file : str, optional
        Path to HTML file to convert to image
    data : pandas.DataFrame, optional
        Data used for the analysis, used to generate HTML if no content/file provided
    results : dict, optional
        Results from the analysis, used to generate HTML if no content/file provided
    figures : dict, optional
        Visualization figures, used to generate HTML if no content/file provided
    target_variable : str, optional
        Name of the target variable, used to generate HTML if no content/file provided
    settings : dict, optional
        Settings for the report
    template_path : str, optional
        Path to a custom HTML template file, used to generate HTML if no content/file provided
    method : str, default 'auto'
        Method to use for image generation ('auto', 'imgkit', 'selenium', 'weasyprint')
    width : int, default 1080
        Width of the image in pixels
    height : int, default 1920
        Height of the image in pixels
    quality : int, default 90
        Image quality (0-100)
        
    Returns
    -------
    str
        Path to the generated image
    """
    return generate_image_report(
        output_path=output_path,
        html_content=html_content,
        html_file=html_file,
        data=data,
        results=results,
        figures=figures,
        target_variable=target_variable,
        settings=settings,
        template_path=template_path,
        method=method,
        orientation='portrait',
        width=width,
        height=height,
        quality=quality,
        **kwargs
    )

__all__ = [
    'generate_image_report',
    'generate_landscape_report',
    'generate_portrait_report',
    'check_dependencies',
    'get_missing_dependencies',
    'install_dependencies'
] 