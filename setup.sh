#!/bin/bash
# ResumeIQ - Quick Setup Script
# Run: bash setup.sh

echo ""
echo "=========================================="
echo "  ResumeIQ — AI Resume Analyzer Setup"
echo "=========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed. Please install Python 3.10+"
    exit 1
fi
echo "✓ Python found: $(python3 --version)"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "Running database migrations..."
python manage.py makemigrations
python manage.py migrate

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  python manage.py runserver"
echo ""
echo "Then open: http://127.0.0.1:8000"
echo ""
