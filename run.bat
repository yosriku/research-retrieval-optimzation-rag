@echo off
REM RAG Application Run Script for Windows
REM This script helps you quickly run the application

echo ==========================================
echo RAG Application with FAISS ^& Gemini
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python is not installed
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo [√] Python found
python --version
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [*] Creating virtual environment...
    python -m venv venv
    echo [√] Virtual environment created
    echo.
)

REM Activate virtual environment
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Check if requirements are installed
if not exist "venv\installed.flag" (
    echo [*] Installing dependencies...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo. > venv\installed.flag
    echo [√] Dependencies installed
    echo.
) else (
    echo [√] Dependencies already installed
    echo.
)

REM Check if vector_dbs folder exists
if not exist "vector_dbs\" (
    echo [!] vector_dbs folder not found
    echo [*] Running setup script...
    python setup_folders.py
    echo.
    echo [!] Please add your database files before running the app
    echo    Required files in each folder:
    echo    - index.faiss
    echo    - index.pkl
    echo.
    pause
)

REM Run the Streamlit app
echo [*] Starting RAG Application...
echo ==========================================
echo.
streamlit run app.py

REM Deactivate virtual environment on exit
call venv\Scripts\deactivate.bat