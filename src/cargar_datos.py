import os
import pandas as pd
def cargarDatos():
    # Obtiene la ruta del directorio donde se encuentra este script (src/)
    script_dir = os.path.dirname(__file__)
    # Sube a la carpeta raiz
    project_dir = os.path.dirname(script_dir)
    # Construye la ruta completa al archivo .xlsx
    file_path = os.path.join(project_dir, 'Base_de_datos.xlsx')
    # Lee el archivo de Excel usando la ruta completa
    df = pd.read_excel(file_path)
    return df