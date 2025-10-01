# Action Recognition in Videos
# Advanced deep learning models for video action classification

## 🎬 Overview

This project implements state-of-the-art action recognition in videos using deep learning models trained on the Kinetics-400 dataset. It supports multiple model architectures and provides both command-line and web-based interfaces for video analysis.

## Features

- **Multiple Model Support**: R3D-18, MC3-18, R(2+1)D-18 architectures
- **Complete Kinetics-400**: Full 400 action class support
- **Automatic Device Selection**: CUDA, MPS (Apple Silicon), or CPU
- **Batch Processing**: Process multiple videos simultaneously
- **Web Interface**: Modern Streamlit-based UI
- **Comprehensive Analytics**: Visualization and statistics
- **Mock Database**: Built-in data storage and management
- **Robust Error Handling**: Graceful failure handling

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Action-Recognition-in-Videos.git
   cd Action-Recognition-in-Videos
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # or run the setup script
   python setup.py
   ```

3. **Run the application**:
   
   **Command Line Interface**:
   ```bash
   python modern_action_recognition.py
   ```
   
   **Web Interface**:
   ```bash
   streamlit run action_recognition_app.py
   ```

## 📁 Project Structure

```
action-recognition-videos/
├── 0146.py                          # Original implementation
├── modern_action_recognition.py     # Modernized CLI version
├── action_recognition_app.py        # Streamlit web application
├── requirements.txt                 # Python dependencies
├── setup.py                        # Setup script
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── data/                           # Data directory
├── output/                         # Output files
├── models/                         # Model storage
└── logs/                           # Log files
```

## Supported Actions

The system recognizes 400 different actions from the Kinetics-400 dataset, including:

- **Sports**: basketball, soccer, tennis, swimming, etc.
- **Daily Activities**: cooking, cleaning, eating, sleeping, etc.
- **Entertainment**: dancing, singing, playing instruments, etc.
- **Work**: typing, presenting, construction, etc.
- **And many more...**

## 🔧 Usage Examples

### Command Line Usage

```python
from modern_action_recognition import ModernActionRecognition

# Initialize the system
recognizer = ModernActionRecognition(model_name='r3d_18', device='auto')

# Load model
recognizer.load_model()

# Process a single video
video_tensor, duration, fps = recognizer.load_video_frames('video.mp4')
predictions = recognizer.predict(video_tensor)

# Process multiple videos
results = recognizer.batch_predict(['video1.mp4', 'video2.mp4'])
```

### Web Interface Usage

1. Start the Streamlit app: `streamlit run action_recognition_app.py`
2. Select your preferred model architecture
3. Upload video files or use sample videos
4. View predictions and analytics
5. Export results

## Model Performance

| Model | Architecture | Parameters | Top-1 Accuracy |
|-------|-------------|------------|----------------|
| R3D-18 | 3D ResNet | ~33M | ~54.5% |
| MC3-18 | Mixed Convolution | ~33M | ~54.0% |
| R(2+1)D-18 | Factorized 3D Conv | ~33M | ~57.5% |

## 🛠️ Technical Details

### Video Preprocessing
- Frame sampling: 16 frames per video
- Resolution: 112x112 pixels
- Normalization: Kinetics-specific mean/std
- Temporal sampling: Evenly distributed across video duration

### Model Architecture
- **R3D-18**: 3D ResNet with 18 layers
- **MC3-18**: Mixed 2D/3D convolutions
- **R(2+1)D-18**: Factorized 3D convolutions

### Device Support
- **CUDA**: NVIDIA GPUs with CUDA support
- **MPS**: Apple Silicon (M1/M2) GPUs
- **CPU**: Fallback for all systems

## Analytics Features

- **Confidence Distribution**: Histogram of prediction confidences
- **Action Frequency**: Most commonly predicted actions
- **Video Statistics**: Duration, FPS, and metadata analysis
- **Batch Results**: Comprehensive multi-video analysis

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure PyTorch is properly installed
   - Check internet connection for model downloads
   - Verify device compatibility

2. **Video Processing Errors**:
   - Supported formats: MP4, AVI, MOV, MKV
   - Check video file integrity
   - Ensure sufficient disk space

3. **Memory Issues**:
   - Reduce batch size for large videos
   - Use CPU if GPU memory is limited
   - Process videos individually

### Performance Tips

- Use GPU acceleration when available
- Process videos in smaller batches
- Use appropriate video resolution
- Monitor system resources

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **TorchVision**: For pretrained video models
- **Kinetics Dataset**: For the comprehensive action recognition dataset
- **Streamlit**: For the web interface framework

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation


# Action-Recognition-in-Videos
