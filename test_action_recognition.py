import unittest
import torch
import torchvision
import numpy as np
import os
import tempfile
import cv2
from modern_action_recognition import ModernActionRecognition

class TestActionRecognition(unittest.TestCase):
    """Test cases for the Action Recognition system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.recognizer = ModernActionRecognition(model_name='r3d_18', device='cpu')
        
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.recognizer.model_name, 'r3d_18')
        self.assertIsNotNone(self.recognizer.classes)
        self.assertEqual(len(self.recognizer.classes), 400)
        
    def test_device_selection(self):
        """Test automatic device selection"""
        device = self.recognizer._get_device('auto')
        self.assertIsInstance(device, torch.device)
        
    def test_kinetics_classes(self):
        """Test Kinetics-400 class loading"""
        classes = self.recognizer._load_kinetics_classes()
        self.assertEqual(len(classes), 400)
        self.assertIn('abseiling', classes)
        self.assertIn('zumba', classes)
        
    def test_transform_creation(self):
        """Test transform creation"""
        transform = self.recognizer._get_transform()
        self.assertIsNotNone(transform)
        
    def test_model_loading(self):
        """Test model loading"""
        # This test might fail if model download fails
        try:
            success = self.recognizer.load_model()
            if success:
                self.assertIsNotNone(self.recognizer.model)
                self.assertEqual(self.recognizer.model.training, False)
        except Exception as e:
            self.skipTest(f"Model loading failed: {e}")
    
    def test_video_creation(self):
        """Test sample video creation"""
        # Create a temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Create a simple test video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, 30.0, (640, 480))
            
            for i in range(30):  # 1 second at 30 fps
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), -1)
                out.write(frame)
            
            out.release()
            
            # Test video loading
            video_tensor, duration, fps = self.recognizer.load_video_frames(temp_path)
            
            self.assertIsNotNone(video_tensor)
            self.assertGreater(duration, 0)
            self.assertGreater(fps, 0)
            self.assertEqual(video_tensor.shape[0], 1)  # Batch size
            self.assertEqual(video_tensor.shape[1], 3)  # Channels
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_prediction_without_model(self):
        """Test prediction without loaded model"""
        # Create dummy video tensor
        dummy_tensor = torch.randn(1, 3, 16, 112, 112)
        
        # Should return empty list without model
        predictions = self.recognizer.predict(dummy_tensor)
        self.assertEqual(predictions, [])
    
    def test_batch_processing_empty_list(self):
        """Test batch processing with empty list"""
        results = self.recognizer.batch_predict([])
        self.assertEqual(results, [])
    
    def test_error_handling_invalid_video(self):
        """Test error handling for invalid video file"""
        video_tensor, duration, fps = self.recognizer.load_video_frames('nonexistent.mp4')
        self.assertIsNone(video_tensor)
        self.assertEqual(duration, 0)
        self.assertEqual(fps, 0)

class TestMockDatabase(unittest.TestCase):
    """Test cases for the mock database functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        from action_recognition_app import MockVideoDatabase
        self.db = MockVideoDatabase()
        
    def test_add_video(self):
        """Test adding video to database"""
        video_id = self.db.add_video('test.mp4', 10.5, 30.0, (1, 3, 16, 112, 112))
        self.assertEqual(video_id, 0)
        self.assertEqual(len(self.db.videos), 1)
        
    def test_add_result(self):
        """Test adding result to database"""
        self.db.add_video('test.mp4', 10.5, 30.0, (1, 3, 16, 112, 112))
        predictions = [{'class': 'test_action', 'confidence': 95.0, 'index': 0}]
        self.db.add_result(0, predictions, 'r3d_18')
        
        self.assertEqual(len(self.db.results), 1)
        self.assertEqual(self.db.results[0]['predictions'], predictions)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
