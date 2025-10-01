# Configuration file for Action Recognition Project

# Model Configuration
MODELS = {
    'r3d_18': {
        'name': 'R3D-18',
        'description': '3D ResNet with 18 layers',
        'parameters': '~33M',
        'accuracy': '~54.5%'
    },
    'mc3_18': {
        'name': 'MC3-18', 
        'description': 'Mixed 2D/3D convolutions',
        'parameters': '~33M',
        'accuracy': '~54.0%'
    },
    'r2plus1d_18': {
        'name': 'R(2+1)D-18',
        'description': 'Factorized 3D convolutions',
        'parameters': '~33M',
        'accuracy': '~57.5%'
    }
}

# Video Processing Configuration
VIDEO_CONFIG = {
    'num_frames': 16,
    'frame_size': (112, 112),
    'supported_formats': ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'],
    'max_duration': 300,  # seconds
    'min_duration': 1,    # seconds
    'max_file_size': 100 * 1024 * 1024  # 100MB
}

# Normalization parameters for Kinetics dataset
NORMALIZATION = {
    'mean': [0.43216, 0.394666, 0.37645],
    'std': [0.22803, 0.22145, 0.216989]
}

# Device Configuration
DEVICE_CONFIG = {
    'auto': 'Automatic selection',
    'cpu': 'CPU only',
    'cuda': 'NVIDIA GPU (CUDA)',
    'mps': 'Apple Silicon (MPS)'
}

# UI Configuration
UI_CONFIG = {
    'page_title': 'Action Recognition in Videos',
    'page_icon': '🎬',
    'layout': 'wide',
    'theme': 'light',
    'sidebar_state': 'expanded'
}

# Database Configuration
DATABASE_CONFIG = {
    'max_videos': 1000,
    'max_results': 10000,
    'cleanup_interval': 24 * 60 * 60,  # 24 hours in seconds
    'export_formats': ['csv', 'json', 'xlsx']
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/action_recognition.log',
    'max_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'batch_size': 1,
    'num_workers': 4,
    'pin_memory': True,
    'prefetch_factor': 2,
    'max_memory_usage': 0.8  # 80% of available memory
}

# Analytics Configuration
ANALYTICS_CONFIG = {
    'confidence_threshold': 0.1,  # 10%
    'top_k_predictions': 5,
    'visualization_dpi': 300,
    'chart_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
}
