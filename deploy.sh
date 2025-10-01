#!/bin/bash

# Action Recognition Deployment Script
# This script helps deploy the action recognition application

set -e

echo "🎬 Action Recognition Deployment Script"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
    else
        print_error "pip3 is not installed. Please install pip."
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        print_success "Dependencies installed successfully"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data output models logs
    print_success "Directories created"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    if [ -f "test_action_recognition.py" ]; then
        python3 -m pytest test_action_recognition.py -v
        print_success "Tests completed"
    else
        print_warning "Test file not found, skipping tests"
    fi
}

# Check PyTorch installation
check_pytorch() {
    print_status "Checking PyTorch installation..."
    python3 -c "
import torch
import torchvision
print(f'PyTorch version: {torch.__version__}')
print(f'TorchVision version: {torchvision.__version__}')
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS (Apple Silicon) available')
else:
    print('Using CPU')
"
    print_success "PyTorch check completed"
}

# Deploy with Docker
deploy_docker() {
    print_status "Deploying with Docker..."
    if command -v docker &> /dev/null; then
        if [ -f "Dockerfile" ]; then
            docker build -t action-recognition .
            print_success "Docker image built successfully"
            
            if [ -f "docker-compose.yml" ]; then
                docker-compose up -d
                print_success "Application deployed with Docker Compose"
                print_status "Access the application at: http://localhost:8501"
            else
                docker run -d -p 8501:8501 --name action-recognition action-recognition
                print_success "Application deployed with Docker"
                print_status "Access the application at: http://localhost:8501"
            fi
        else
            print_error "Dockerfile not found"
            exit 1
        fi
    else
        print_error "Docker is not installed"
        exit 1
    fi
}

# Deploy locally
deploy_local() {
    print_status "Deploying locally..."
    print_status "Starting Streamlit application..."
    print_status "Access the application at: http://localhost:8501"
    streamlit run action_recognition_app.py
}

# Main deployment function
main() {
    local deployment_type=${1:-local}
    
    case $deployment_type in
        "docker")
            check_python
            check_pip
            install_dependencies
            create_directories
            run_tests
            check_pytorch
            deploy_docker
            ;;
        "local")
            check_python
            check_pip
            install_dependencies
            create_directories
            run_tests
            check_pytorch
            deploy_local
            ;;
        "setup")
            check_python
            check_pip
            install_dependencies
            create_directories
            run_tests
            check_pytorch
            print_success "Setup completed successfully!"
            print_status "You can now run:"
            print_status "  python3 modern_action_recognition.py"
            print_status "  streamlit run action_recognition_app.py"
            ;;
        *)
            echo "Usage: $0 {local|docker|setup}"
            echo "  local  - Deploy locally with Streamlit"
            echo "  docker - Deploy with Docker"
            echo "  setup  - Setup environment only"
            exit 1
            ;;
    esac
}

# Run main function with arguments
main "$@"
