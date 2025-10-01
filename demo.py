#!/usr/bin/env python3
"""
Action Recognition Demo Script
Demonstrates all the modern features of the action recognition system
"""

import os
import sys
import time
from modern_action_recognition import ModernActionRecognition, create_sample_videos

def print_banner():
    """Print a nice banner"""
    print("🎬" + "="*60 + "🎬")
    print("    ADVANCED ACTION RECOGNITION IN VIDEOS DEMO")
    print("🎬" + "="*60 + "🎬")
    print()

def demo_model_loading():
    """Demonstrate model loading capabilities"""
    print("🔧 MODEL LOADING DEMONSTRATION")
    print("-" * 40)
    
    models = ['r3d_18', 'mc3_18', 'r2plus1d_18']
    
    for model_name in models:
        print(f"\n📦 Testing {model_name}...")
        recognizer = ModernActionRecognition(model_name=model_name, device='auto')
        
        if recognizer.load_model():
            print(f"✅ {model_name} loaded successfully!")
            print(f"   Device: {recognizer.device}")
            print(f"   Classes: {len(recognizer.classes)}")
        else:
            print(f"❌ Failed to load {model_name}")
    
    print()

def demo_sample_video_creation():
    """Demonstrate sample video creation"""
    print("🎥 SAMPLE VIDEO CREATION DEMONSTRATION")
    print("-" * 40)
    
    print("Creating sample videos...")
    sample_videos = create_sample_videos()
    
    print(f"✅ Created {len(sample_videos)} sample videos:")
    for video in sample_videos:
        if os.path.exists(video):
            size = os.path.getsize(video) / 1024  # KB
            print(f"   📹 {video} ({size:.1f} KB)")
    
    print()
    return sample_videos

def demo_video_processing(sample_videos):
    """Demonstrate video processing capabilities"""
    print("🔍 VIDEO PROCESSING DEMONSTRATION")
    print("-" * 40)
    
    recognizer = ModernActionRecognition(model_name='r3d_18', device='auto')
    
    if not recognizer.load_model():
        print("❌ Failed to load model for processing demo")
        return
    
    print("Processing sample videos...")
    results = recognizer.batch_predict(sample_videos)
    
    print(f"\n📊 PROCESSING RESULTS:")
    for i, result in enumerate(results):
        print(f"\n🎬 Video {i+1}: {result['video_path']}")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"   FPS: {result['fps']:.1f}")
        print("   Top Predictions:")
        
        for j, pred in enumerate(result['predictions'][:3]):
            print(f"   {j+1}. {pred['class'].replace('_', ' ').title()}: {pred['confidence']:.1f}%")
    
    print()
    return results

def demo_analytics(results):
    """Demonstrate analytics capabilities"""
    print("📈 ANALYTICS DEMONSTRATION")
    print("-" * 40)
    
    if not results:
        print("No results to analyze")
        return
    
    recognizer = ModernActionRecognition()
    
    print("Creating analytics visualizations...")
    recognizer.visualize_predictions(results)
    
    # Calculate some basic statistics
    total_videos = len(results)
    total_duration = sum(r['duration'] for r in results)
    avg_confidence = sum(pred['confidence'] for r in results for pred in r['predictions']) / len([pred for r in results for pred in r['predictions']])
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total Videos Processed: {total_videos}")
    print(f"   Total Duration: {total_duration:.2f}s")
    print(f"   Average Confidence: {avg_confidence:.1f}%")
    
    # Most predicted actions
    all_predictions = [pred['class'] for r in results for pred in r['predictions']]
    from collections import Counter
    top_actions = Counter(all_predictions).most_common(3)
    
    print(f"   Most Predicted Actions:")
    for action, count in top_actions:
        print(f"     - {action.replace('_', ' ').title()}: {count} times")
    
    print()

def demo_web_interface():
    """Demonstrate web interface capabilities"""
    print("🌐 WEB INTERFACE DEMONSTRATION")
    print("-" * 40)
    
    print("The web interface provides:")
    print("   🎥 Single Video Analysis - Upload and analyze individual videos")
    print("   📊 Batch Processing - Process multiple videos simultaneously")
    print("   📈 Analytics Dashboard - View statistics and visualizations")
    print("   🗄️ Database Management - Store and export results")
    print("   📋 Sample Videos - Generate test videos for demonstration")
    
    print("\n🚀 To start the web interface, run:")
    print("   streamlit run action_recognition_app.py")
    print("   Then open http://localhost:8501 in your browser")
    
    print()

def demo_deployment_options():
    """Demonstrate deployment options"""
    print("🚀 DEPLOYMENT OPTIONS DEMONSTRATION")
    print("-" * 40)
    
    print("Available deployment methods:")
    print("   1. Local Deployment:")
    print("      ./deploy.sh local")
    print("      streamlit run action_recognition_app.py")
    
    print("\n   2. Docker Deployment:")
    print("      ./deploy.sh docker")
    print("      docker-compose up -d")
    
    print("\n   3. Setup Only:")
    print("      ./deploy.sh setup")
    print("      python3 modern_action_recognition.py")
    
    print("\n   4. Manual Setup:")
    print("      pip install -r requirements.txt")
    print("      python3 setup.py")
    
    print()

def cleanup_demo_files(sample_videos):
    """Clean up demo files"""
    print("🧹 CLEANING UP DEMO FILES")
    print("-" * 40)
    
    cleaned_count = 0
    for video in sample_videos:
        if os.path.exists(video):
            os.remove(video)
            cleaned_count += 1
    
    print(f"✅ Cleaned up {cleaned_count} sample video files")
    
    # Clean up analysis image if it exists
    if os.path.exists('action_recognition_analysis.png'):
        os.remove('action_recognition_analysis.png')
        print("✅ Cleaned up analysis visualization")
    
    print()

def main():
    """Main demo function"""
    print_banner()
    
    try:
        # Demo 1: Model Loading
        demo_model_loading()
        
        # Demo 2: Sample Video Creation
        sample_videos = demo_sample_video_creation()
        
        # Demo 3: Video Processing
        results = demo_video_processing(sample_videos)
        
        # Demo 4: Analytics
        demo_analytics(results)
        
        # Demo 5: Web Interface
        demo_web_interface()
        
        # Demo 6: Deployment Options
        demo_deployment_options()
        
        # Cleanup
        cleanup_demo_files(sample_videos)
        
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The action recognition system is ready for use!")
        print("Check the PROJECT_SUMMARY.md for complete details.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
        cleanup_demo_files(sample_videos if 'sample_videos' in locals() else [])
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        cleanup_demo_files(sample_videos if 'sample_videos' in locals() else [])

if __name__ == "__main__":
    main()
