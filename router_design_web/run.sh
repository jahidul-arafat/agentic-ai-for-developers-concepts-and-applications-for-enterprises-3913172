#!/bin/bash

echo "🚀 Agent Router Web Interface Startup Script"
echo "=============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads
mkdir -p logs

# Set environment variables if not set
export FLASK_ENV=development
export FLASK_DEBUG=1

# Start the application
echo "🚀 Starting Agent Router Web Interface..."
echo "💻 Frontend will be available at: http://localhost:5010"
echo "📚 API documentation at: http://localhost:5010/api/docs"
echo "🔍 Health check at: http://localhost:5010/api/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=============================================="

python3 app.py