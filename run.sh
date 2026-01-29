#!/bin/bash

# RAG Application Run Script
# This script helps you quickly run the application

echo "=========================================="
echo "RAG Application with FAISS & Gemini"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo ""

# Check if requirements are installed
if [ ! -f "venv/installed.flag" ]; then
    echo "📥 Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    touch venv/installed.flag
    echo "✅ Dependencies installed"
    echo ""
else
    echo "✅ Dependencies already installed"
    echo ""
fi

# Check if vector_dbs folder exists
if [ ! -d "vector_dbs" ]; then
    echo "⚠️  vector_dbs folder not found"
    echo "🔧 Running setup script..."
    python3 setup_folders.py
    echo ""
    echo "⚠️  Please add your database files before running the app"
    echo "   Required files in each folder:"
    echo "   - index.faiss"
    echo "   - index.pkl"
    echo ""
    read -p "Press Enter to continue or Ctrl+C to exit..."
fi

# Run the Streamlit app
echo "🚀 Starting RAG Application..."
echo "=========================================="
echo ""
streamlit run app.py

# Deactivate virtual environment on exit
deactivate