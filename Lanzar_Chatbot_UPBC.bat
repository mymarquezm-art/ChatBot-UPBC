@echo off
title Lanzador Maestro - Chatbot UPBC (Backup + Limpieza)
cls

:: --- CONFIGURACIÓN DE RUTAS ---
set PY_EXE="C:\Users\UPBC_Misabel\AppData\Local\Programs\Python\Python310\python.exe"
set VENV_DIR=D:\AI_ENV
set APP_FOLDER=D:\c_google\santander\M16
set DB_DIR=D:\c_google\santander\M16\chroma_db_hybrid
set BACKUP_DIR=D:\c_google\santander\M16\backup_db
set TEMP_DIR=D:\AI_TEMP
set HF_HOME=D:\AI_TEMP\huggingface

echo ======================================================
echo    CONFIGURANDO ENTORNO AUTONOMO (PARTICION D)
echo ======================================================

:: 1. Crear estructura de carpetas si no existen
if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

:: 2. Crear/Verificar Entorno Virtual
if not exist "%VENV_DIR%" (
    echo [+] Creando entorno virtual en %VENV_DIR%...
    %PY_EXE% -m venv "%VENV_DIR%"
)

:: 3. Activar Entorno y configurar Variables de Entorno
echo [+] Activando entorno y configurando variables...
call "%VENV_DIR%\Scripts\activate"
set USERPROFILE=%TEMP_DIR%
set HF_HOME=%HF_HOME%

:: 4. Gestión de Dependencias (Según tus especificaciones)
echo [+] Verificando librerias criticas...
pip install streamlit langchain-google-genai langchain-community pypdf chromadb langchain-huggingface plotly tenacity >nul 2>&1

echo [+] Aplicando parches de compatibilidad (Sentence-Transformers 2.7.0)...
pip install "huggingface-hub<1.0.0" --quiet
pip install --upgrade langchain-huggingface transformers huggingface-hub accelerator --quiet
pip uninstall -y sentence-transformers >nul 2>&1
pip install sentence-transformers==2.7.0 --quiet

:: 5. Lanzamiento de la Aplicación
echo ======================================================
echo    INICIANDO INTERFAZ... (Cierra la ventana para limpiar)
echo ======================================================
d:
cd "%APP_FOLDER%"

:: Ejecutar Streamlit
streamlit run castone_geminiH_v15.py

:: 6. Proceso de Backup y Limpieza (Se ejecuta al cerrar Streamlit)
echo.
echo ======================================================
echo    REALIZANDO BACKUP Y LIMPIANDO ARCHIVOS TEMPORALES
echo ======================================================

:: Copia de seguridad de la Base de Datos
if exist "%DB_DIR%" (
    echo [+] Respaldando base de datos vectorial...
    xcopy "%DB_DIR%" "%BACKUP_DIR%" /E /I /Y >nul
)

deactivate
timeout /t 3 >nul

:: Limpieza profunda de temporales
echo [+] Borrando cache de modelos en D:\AI_TEMP...
del /s /q "%TEMP_DIR%\*" >nul 2>&1
for /d %%p in ("%TEMP_DIR%\*") do rd /s /q "%%p" >nul 2>&1

echo [+] Proceso finalizado exitosamente.
timeout /t 2 >nul
exit