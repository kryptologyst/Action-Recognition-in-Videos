# Action Recognition in Videos - Setup Script
# This script sets up the environment and installs dependencies

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = ["data", "output", "models", "logs"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def check_pytorch():
    """Check PyTorch installation and device availability"""
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("🍎 MPS (Apple Silicon) available")
        else:
            print("💻 Using CPU")
            
    except ImportError:
        print("❌ PyTorch not installed properly")
        return False
    return True

def main():
    """Main setup function"""
    print("🎬 Action Recognition Setup")
    print("=" * 40)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("Setup failed. Please check error messages above.")
        return
    
    # Check PyTorch
    if not check_pytorch():
        print("PyTorch check failed. Please reinstall PyTorch.")
        return
    
    print("\n🎉 Setup complete!")
    print("You can now run:")
    print("  python modern_action_recognition.py")
    print("  streamlit run action_recognition_app.py")

if __name__ == "__main__":
    main()
