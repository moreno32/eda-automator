"""
Generador de informes en formato de imagen.

Este módulo contiene funciones para generar informes
como imágenes en formato apaisado (landscape) o vertical (portrait).
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil

def generate_landscape_report(output_path, html_path, eda):
    """
    Genera un informe en formato de imagen apaisada a partir de un HTML.
    
    Args:
        output_path (str): Ruta donde se guardará la imagen
        html_path (str): Ruta al archivo HTML para convertir a imagen
        eda (EDAAutomator): Instancia de EDA con los resultados
        
    Returns:
        str: Ruta a la imagen generada o None si falla
    """
    print("\nGenerando informe en formato imagen (apaisado)...")
    
    # Asegurar que el directorio de salida exista
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Intentar diferentes métodos para generar imagen
    result = None
    
    # Método 1: Usar imgkit si está disponible
    try:
        import imgkit
        print("Usando imgkit para generar imagen...")
        options = {
            'format': 'png',
            'width': 1200,
            'quality': 100,
            'encoding': 'UTF-8',
            'enable-local-file-access': None,
            'quiet': None
        }
        result = imgkit.from_file(html_path, output_path, options=options)
        print(f"Imagen generada correctamente en {output_path}")
        return output_path
    except (ImportError, Exception) as e:
        print(f"Error al usar imgkit: {str(e)}")
    
    # Método 2: Usar selenium si está disponible
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        print("Usando selenium para generar imagen...")
        
        # Configurar opciones de Chrome
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1200,800')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        # Iniciar navegador
        driver = webdriver.Chrome(options=chrome_options)
        
        # Cargar archivo HTML
        html_path_url = f"file:///{os.path.abspath(html_path)}"
        driver.get(html_path_url)
        
        # Tomar screenshot
        driver.save_screenshot(output_path)
        driver.quit()
        
        print(f"Imagen generada correctamente en {output_path}")
        return output_path
    except (ImportError, Exception) as e:
        print(f"Error al usar selenium: {str(e)}")
    
    # Método 3: Usar weasyprint si está disponible
    try:
        from weasyprint import HTML
        import pdf2image
        print("Usando weasyprint para generar imagen...")
        
        # Generar PDF primero
        pdf_path = output_path.replace('.png', '.pdf')
        HTML(filename=html_path).write_pdf(pdf_path)
        
        # Convertir PDF a imagen
        images = pdf2image.convert_from_path(pdf_path, dpi=100)
        images[0].save(output_path, 'PNG')
        
        # Eliminar PDF temporal
        os.remove(pdf_path)
        
        print(f"Imagen generada correctamente en {output_path}")
        return output_path
    except (ImportError, Exception) as e:
        print(f"Error al usar weasyprint/pdf2image: {str(e)}")
    
    # Si todo falla, notificar al usuario
    print("No se pudo generar la imagen del informe. Instale alguna de las siguientes dependencias:")
    print("  - imgkit + wkhtmltopdf")
    print("  - selenium + webdriver para Chrome")
    print("  - weasyprint + pdf2image")
    
    return None

def generate_portrait_report(output_path, html_path, eda):
    """
    Genera un informe en formato de imagen vertical a partir de un HTML.
    
    Args:
        output_path (str): Ruta donde se guardará la imagen
        html_path (str): Ruta al archivo HTML para convertir a imagen
        eda (EDAAutomator): Instancia de EDA con los resultados
        
    Returns:
        str: Ruta a la imagen generada o None si falla
    """
    print("\nGenerando informe en formato imagen (vertical)...")
    
    # Asegurar que el directorio de salida exista
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Intentar diferentes métodos para generar imagen vertical
    result = None
    
    # Método 1: Usar imgkit si está disponible
    try:
        import imgkit
        print("Usando imgkit para generar imagen vertical...")
        options = {
            'format': 'png',
            'width': 800,  # Ancho más estrecho para formato vertical
            'quality': 100,
            'encoding': 'UTF-8',
            'enable-local-file-access': None,
            'quiet': None
        }
        result = imgkit.from_file(html_path, output_path, options=options)
        print(f"Imagen vertical generada correctamente en {output_path}")
        return output_path
    except (ImportError, Exception) as e:
        print(f"Error al usar imgkit: {str(e)}")
    
    # Método 2: Usar selenium si está disponible
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        import time
        print("Usando selenium para generar imagen vertical...")
        
        # Configurar opciones de Chrome
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=800,30000')  # Alto muy grande para capturar todo
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        # Iniciar navegador
        driver = webdriver.Chrome(options=chrome_options)
        
        # Cargar archivo HTML
        html_path_url = f"file:///{os.path.abspath(html_path)}"
        driver.get(html_path_url)
        
        # Esperar a que se cargue el contenido
        time.sleep(2)
        
        # Obtener la altura real del contenido
        height = driver.execute_script("return document.body.scrollHeight")
        
        # Ajustar el tamaño de la ventana para capturar todo el contenido
        driver.set_window_size(800, height + 100)
        
        # Tomar screenshot
        driver.save_screenshot(output_path)
        driver.quit()
        
        print(f"Imagen vertical generada correctamente en {output_path}")
        return output_path
    except (ImportError, Exception) as e:
        print(f"Error al usar selenium: {str(e)}")
    
    # Método 3: Usar weasyprint si está disponible
    try:
        from weasyprint import HTML
        import pdf2image
        print("Usando weasyprint para generar imagen vertical...")
        
        # Generar PDF primero
        pdf_path = output_path.replace('.png', '.pdf')
        HTML(filename=html_path).write_pdf(pdf_path)
        
        # Convertir PDF a imagen
        images = pdf2image.convert_from_path(pdf_path, dpi=100)
        images[0].save(output_path, 'PNG')
        
        # Eliminar PDF temporal
        os.remove(pdf_path)
        
        print(f"Imagen vertical generada correctamente en {output_path}")
        return output_path
    except (ImportError, Exception) as e:
        print(f"Error al usar weasyprint/pdf2image: {str(e)}")
    
    # Si todo falla, notificar al usuario
    print("No se pudo generar la imagen vertical del informe. Instale alguna de las siguientes dependencias:")
    print("  - imgkit + wkhtmltopdf")
    print("  - selenium + webdriver para Chrome")
    print("  - weasyprint + pdf2image")
    
    return None 