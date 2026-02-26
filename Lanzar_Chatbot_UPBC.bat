@echo off
import subprocess
import sys
import os
import shutil
import time

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Directorio base del proyecto
BASE_DIR = r"C:\Asistente_UPBC"
# Nombre de tu archivo principal (debe estar en BASE_DIR)
ARCHIVO_APP = "streamlit_app.py"

# Entorno Virtual y Temporales
VENV_DIR = os.path.join(BASE_DIR, "env_chatbot")
TEMP_DIR = r"C:\AI_TEMP"
HF_HOME = os.path.join(TEMP_DIR, "huggingface")
BACKUP_DIR = os.path.join(BASE_DIR, "backups_db")
CHROMA_DB = os.path.join(BASE_DIR, "chroma_db_hybrid")

REQUERIMIENTOS = [
    "streamlit", "langchain", "langchain-community", 
    "langchain-google-genai", "pypdf", "chromadb", 
    "pandas", "plotly", "tenacity", "google-generativeai",
    "langchain-huggingface", "transformers", "accelerate",
    "sentence-transformers==2.7.0" # Versión específica requerida
]

def inicializar_directorios():
    """Crea la estructura de carpetas si no existe."""
    for folder in [BASE_DIR, TEMP_DIR, HF_HOME, BACKUP_DIR]:
        if not os.path.exists(folder):
            print(f"[+] Creando directorio: {folder}")
            os.makedirs(folder)

def gestionar_entorno_virtual():
    """Crea el entorno virtual si no existe y devuelve la ruta del ejecutable python."""
    python_venv = os.path.join(VENV_DIR, "Scripts", "python.exe")
    
    if not os.path.exists(VENV_DIR):
        print("--- No se encontró entorno virtual. Creando uno nuevo... ---")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
        
    return python_venv

def instalar_dependencias(python_venv):
    """Instala las librerías necesarias dentro del entorno virtual."""
    print("--- Verificando e instalando librerías necesarias... ---")
    try:
        subprocess.check_call([python_venv, "-m", "pip", "install", "--upgrade", "pip"], stdout=subprocess.DEVNULL)
        subprocess.check_call([python_venv, "-m", "pip", "install", *REQUERIMIENTOS, "--quiet"])
        print("[OK] Librerías listas.")
    except Exception as e:
        print(f"[!] Error instalando dependencias: {e}")

def realizar_backup():
    """Respalda la base de datos antes de salir."""
    if os.path.exists(CHROMA_DB):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        destino = os.path.join(BACKUP_DIR, f"backup_{timestamp}")
        print(f"--- Realizando backup de la base de datos en: {destino} ---")
        shutil.copytree(CHROMA_DB, destino)
    else:
        print("--- No se encontró base de datos para respaldar. ---")

def limpieza_profunda():
    """Elimina archivos temporales y caché de modelos."""
    print("--- Iniciando limpieza profunda de temporales en C: ---")
    time.sleep(3) # Espera a que los procesos suelten los archivos
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        os.makedirs(TEMP_DIR) # Recrea la carpeta vacía para el próximo uso
    print("[OK] Limpieza finalizada.")

def lanzar_app(python_venv):
    """Ejecuta la aplicación usando Streamlit del entorno virtual."""
    streamlit_exe = os.path.join(VENV_DIR, "Scripts", "streamlit.exe")
    app_path = os.path.join(BASE_DIR, ARCHIVO_APP)
    
    if os.path.exists(app_path):
        # Configurar variable de entorno para Hugging Face
        env = os.environ.copy()
        env["HF_HOME"] = HF_HOME
        
        print("--- Lanzando Chatbot UPBC ---")
        try:
            subprocess.run([streamlit_exe, "run", app_path], env=env)
        except KeyboardInterrupt:
            pass
        finally:
            realizar_backup()
            limpieza_profunda()
    else:
        print(f"❌ ERROR: Coloca el archivo {ARCHIVO_APP} en {BASE_DIR}")

if __name__ == "__main__":
    try:
        inicializar_directorios()
        py_env = gestionar_entorno_virtual()
        instalar_dependencias(py_env)
        lanzar_app(py_env)
    except Exception as e:
        print(f"ERROR CRÍTICO: {e}")
        input("Presiona Enter para salir...")

