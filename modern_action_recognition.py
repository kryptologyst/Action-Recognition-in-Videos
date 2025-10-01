# Project 146: Modern Action Recognition in Videos
# Description: Advanced action recognition using state-of-the-art deep learning models
# Features: Multiple model support, batch processing, comprehensive visualization

import torch
import torchvision
import torchvision.transforms as T
import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ModernActionRecognition:
    """Modern action recognition system with multiple model support"""
    
    def __init__(self, model_name='r3d_18', device='auto'):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.classes = self._load_kinetics_classes()
        self.transform = self._get_transform()
        
    def _get_device(self, device):
        """Automatically select the best available device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _load_kinetics_classes(self):
        """Load complete Kinetics-400 class labels (400 classes)"""
        kinetics_classes = [
            "abseiling", "air drumming", "answering questions", "applauding", "applying cream",
            "archery", "arm wrestling", "arranging flowers", "assembling computer", "auctioning",
            "baby waking up", "baking cookies", "balloon blowing", "bandaging", "barbequing",
            "bartending", "beatboxing", "bee keeping", "belly dancing", "bending back",
            "bending metal", "biking through snow", "blasting sand", "blowing glass", "blowing leaves",
            "blowing nose", "blowing out candles", "bobsledding", "bookbinding", "bouncing on trampoline",
            "bowling", "braiding hair", "breading or breadcrumbing", "breakdancing", "breaking boards",
            "breaking glass", "breathing fire", "brush painting", "brushing hair", "brushing teeth",
            "building cabinet", "building lego", "building sandcastle", "building snowman", "bulldozing",
            "bungee jumping", "busking", "canoeing or kayaking", "capoeira", "carrying baby",
            "cartwheeling", "carving pumpkin", "catching fish", "catching or throwing baseball",
            "catching or throwing frisbee", "catching or throwing softball", "celebrating",
            "changing oil", "changing wheel", "checking tires", "chopping wood", "clapping",
            "clay pottery making", "clean and jerk", "cleaning floor", "cleaning gutters",
            "cleaning pool", "cleaning shoes", "cleaning toilet", "cleaning windows", "climbing a rope",
            "climbing ladder", "climbing tree", "contact juggling", "cooking chicken", "cooking egg",
            "cooking on campfire", "cooking sausages", "counting money", "country line dancing",
            "cracking neck", "crawling baby", "crossing river", "crying", "curling hair",
            "cutting nails", "cutting pineapple", "cutting watermelon", "dancing ballet",
            "dancing charleston", "dancing gangnam style", "dancing macarena", "deadlifting",
            "decorating the christmas tree", "digging", "dining", "disc golfing", "diving cliff",
            "docking boat", "doing aerobics", "doing laundry", "doing nails", "drawing",
            "dribbling basketball", "drinking", "drinking beer", "drinking shots", "driving car",
            "driving tractor", "drop kicking", "drumming fingers", "dunking basketball",
            "dying hair", "eating burger", "eating cake", "eating carrots", "eating chips",
            "eating doughnuts", "eating hotdog", "eating ice cream", "eating spaghetti",
            "eating watermelon", "egg hunting", "exercising arm", "exercising with an exercise ball",
            "extinguishing fire", "faceplanting", "feeding birds", "feeding fish", "feeding goats",
            "filling eyebrows", "finger snapping", "fire breathing", "flipping pancake",
            "fly tying", "flying kite", "folding clothes", "folding napkins", "folding paper",
            "front raises", "frying vegetables", "garbage collecting", "gargling", "getting a haircut",
            "getting a tattoo", "giving or receiving award", "golf chipping", "golf driving",
            "golf putting", "grinding meat", "grooming dog", "grooming horse", "gymnastics tumbling",
            "hammer throw", "headbanging", "headbutting", "high jump", "high kick", "hitting baseball",
            "hockey stop", "holding snake", "hopscotch", "hoverboarding", "hugging", "hula hooping",
            "hurling (sport)", "ice climbing", "ice fishing", "ice skating", "ironing", "javelin throw",
            "jetskiing", "jogging", "juggling balls", "juggling fire", "juggling soccer ball",
            "jumping into pool", "jumpstyle dancing", "kicking field goal", "kicking soccer ball",
            "kissing", "kitesurfing", "knitting", "krumping", "laughing", "laying bricks",
            "long jump", "lunge", "making a cake", "making a sandwich", "making bed",
            "making jewelry", "making pizza", "making snowman", "making sushi", "making tea",
            "marching", "massaging back", "massaging feet", "massaging legs", "massaging person's head",
            "metal detecting", "milking cow", "mopping floor", "motorcycling", "moving furniture",
            "mowing lawn", "news anchoring", "opening bottle", "opening present", "paragliding",
            "parasailing", "parkour", "passing American football (in game)", "passing American football (not in game)",
            "peeling apples", "peeling potatoes", "petting animal (not cat)", "petting cat",
            "picking fruit", "planting trees", "plastering", "playing accordion", "playing badminton",
            "playing bagpipes", "playing basketball", "playing bass guitar", "playing cards",
            "playing cello", "playing chess", "playing clarinet", "playing controller", "playing cricket",
            "playing cymbals", "playing didgeridoo", "playing drums", "playing flute",
            "playing guitar", "playing harmonica", "playing harp", "playing ice hockey",
            "playing keyboard", "playing kickball", "playing monopoly", "playing organ",
            "playing paintball", "playing piano", "playing poker", "playing recorder",
            "playing saxophone", "playing squash or racquetball", "playing tennis",
            "playing trombone", "playing trumpet", "playing ukulele", "playing violin",
            "playing volleyball", "playing xylophone", "pole vault", "presenting weather forecast",
            "pull ups", "pumping fist", "pumping gas", "punching bag", "punching person (boxing)",
            "push up", "pushing car", "pushing cart", "pushing wheelchair", "putting in contact lens",
            "raising eyebrows", "reading", "reading newspaper", "recording music", "riding a bike",
            "riding camel", "riding elephant", "riding mechanical bull", "riding mountain bike",
            "riding mule", "riding or walking with horse", "riding scooter", "riding unicycle",
            "ripping paper", "robot dancing", "rock climbing", "rock scissors paper", "roller skating",
            "running on treadmill", "sailing", "salsa dancing", "sanding floor", "sanding wood",
            "sawing", "scrambling eggs", "scuba diving", "setting table", "shaking hands",
            "shaking head", "sharpening knives", "sharpening pencil", "shaving head", "shaving legs",
            "shining shoes", "shooting basketball", "shooting goal (soccer)", "shot put",
            "shoveling snow", "shredding paper", "shuffling cards", "side kick", "sign language interpreting",
            "singing", "situp", "skateboarding", "ski jumping", "skiing (not slalom or crosscountry)",
            "skiing crosscountry", "skiing slalom", "skipping rope", "skydiving", "slacklining",
            "slapping", "sled dog racing", "smoking", "smoking hookah", "snatch weight lifting",
            "sneezing", "sniffing", "snorkeling", "snowboarding", "snowkiting", "snowmobiling",
            "somersaulting", "spinning poi", "spray painting", "spraying", "springboard diving",
            "squat", "sticking tongue out", "stomping grapes", "stretching arm", "stretching leg",
            "strumming guitar", "surfing crowd", "surfing water", "sweeping floor", "swimming backstroke",
            "swimming breast stroke", "swimming butterfly stroke", "swing dancing", "swinging legs",
            "swinging on something", "sword fighting", "tai chi", "taking a shower", "tango dancing",
            "tap dancing", "tapping guitar", "tapping pen", "tasting beer", "tasting food",
            "testifying", "texting", "throwing axe", "throwing ball", "throwing discus",
            "tickling", "tobogganing", "tossing coin", "tossing salad", "training dog",
            "trapezing", "trimming or shaving beard", "trimming trees", "triple jump",
            "tying bow tie", "tying knot (not on a tie)", "tying tie", "unboxing",
            "unloading truck", "using computer", "using remote controller (not gaming)",
            "using segway", "vault", "waiting in line", "walking the dog", "washing dishes",
            "washing feet", "washing hair", "washing hands", "water skiing", "water sliding",
            "watering plants", "waxing car", "waxing chest", "waxing legs", "weaving basket",
            "welding", "whistling", "windsurfing", "wrapping present", "wrestling",
            "writing", "yawning", "yoga", "zumba"
        ]
        return kinetics_classes
    
    def _get_transform(self):
        """Get video preprocessing transforms"""
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                       std=[0.22803, 0.22145, 0.216989])
        ])
    
    def load_model(self):
        """Load the specified model with error handling"""
        try:
            if self.model_name == 'r3d_18':
                self.model = torchvision.models.video.r3d_18(pretrained=True)
            elif self.model_name == 'mc3_18':
                self.model = torchvision.models.video.mc3_18(pretrained=True)
            elif self.model_name == 'r2plus1d_18':
                self.model = torchvision.models.video.r2plus1d_18(pretrained=True)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ {self.model_name} loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_video_frames(self, video_path, num_frames=16, size=(112, 112)):
        """Load and preprocess video frames with robust error handling"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Sample frames evenly across the video
            if total_frames <= num_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            
            cap.release()
            
            if len(frames) == 0:
                raise ValueError("No frames could be extracted from video")
            
            # Pad frames if necessary
            while len(frames) < num_frames:
                frames.append(frames[-1])
            
            # Transform and normalize
            frames = [self.transform(frame) for frame in frames[:num_frames]]
            video_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # [C, T, H, W]
            return video_tensor.unsqueeze(0), duration, fps  # [1, C, T, H, W]
            
        except Exception as e:
            print(f"❌ Error loading video: {e}")
            return None, 0, 0
    
    def predict(self, video_tensor):
        """Make prediction on video tensor"""
        try:
            video_tensor = video_tensor.to(self.device)
            with torch.no_grad():
                outputs = self.model(video_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                top5_indices = torch.topk(probs, k=5).indices
                top5_probs = torch.topk(probs, k=5).values
            
            results = []
            for idx, prob in zip(top5_indices, top5_probs):
                results.append({
                    'class': self.classes[idx.item()],
                    'confidence': prob.item() * 100,
                    'index': idx.item()
                })
            
            return results
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return []
    
    def batch_predict(self, video_paths):
        """Process multiple videos in batch"""
        results = []
        for video_path in tqdm(video_paths, desc="Processing videos"):
            video_tensor, duration, fps = self.load_video_frames(video_path)
            if video_tensor is not None:
                predictions = self.predict(video_tensor)
                results.append({
                    'video_path': video_path,
                    'duration': duration,
                    'fps': fps,
                    'predictions': predictions
                })
        return results
    
    def visualize_predictions(self, results):
        """Create visualization of prediction results"""
        if not results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Action Recognition Analysis', fontsize=16)
        
        # Top predictions
        top_classes = [r['predictions'][0]['class'] for r in results if r['predictions']]
        class_counts = pd.Series(top_classes).value_counts().head(10)
        
        axes[0, 0].barh(range(len(class_counts)), class_counts.values)
        axes[0, 0].set_yticks(range(len(class_counts)))
        axes[0, 0].set_yticklabels(class_counts.index)
        axes[0, 0].set_title('Most Predicted Actions')
        axes[0, 0].set_xlabel('Count')
        
        # Confidence distribution
        all_confidences = []
        for result in results:
            for pred in result['predictions']:
                all_confidences.append(pred['confidence'])
        
        axes[0, 1].hist(all_confidences, bins=20, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Confidence Distribution')
        axes[0, 1].set_xlabel('Confidence (%)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Video duration distribution
        durations = [r['duration'] for r in results]
        axes[1, 0].hist(durations, bins=15, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Video Duration Distribution')
        axes[1, 0].set_xlabel('Duration (seconds)')
        axes[1, 0].set_ylabel('Frequency')
        
        # FPS distribution
        fps_values = [r['fps'] for r in results]
        axes[1, 1].hist(fps_values, bins=15, alpha=0.7, color='orange')
        axes[1, 1].set_title('FPS Distribution')
        axes[1, 1].set_xlabel('Frames Per Second')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('action_recognition_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_sample_videos():
    """Create sample videos for testing"""
    print("🎬 Creating sample videos for demonstration...")
    
    sample_videos = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    actions = ["red_action", "green_action", "blue_action", "yellow_action", "purple_action"]
    
    for i, (color, action) in enumerate(zip(colors, actions)):
        filename = f"sample_{action}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))
        
        for frame_num in range(90):  # 3 seconds at 30 fps
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Create a moving rectangle
            x = int(50 + (frame_num * 5) % 540)
            y = int(50 + (frame_num * 3) % 380)
            cv2.rectangle(frame, (x, y), (x+100, y+100), color, -1)
            cv2.putText(frame, action, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(frame)
        
        out.release()
        sample_videos.append(filename)
    
    print(f"✅ Created {len(sample_videos)} sample videos!")
    return sample_videos

def main():
    """Main function demonstrating the modern action recognition system"""
    print("🎬 Modern Action Recognition in Videos")
    print("=" * 50)
    
    # Initialize the system
    recognizer = ModernActionRecognition(model_name='r3d_18', device='auto')
    
    # Load model
    if not recognizer.load_model():
        print("Failed to load model. Exiting...")
        return
    
    # Create sample videos
    sample_videos = create_sample_videos()
    
    # Process videos
    print("\n🔍 Processing sample videos...")
    results = recognizer.batch_predict(sample_videos)
    
    # Display results
    print("\n📊 Results:")
    for result in results:
        print(f"\n📹 {result['video_path']}")
        print(f"   Duration: {result['duration']:.2f}s, FPS: {result['fps']:.1f}")
        print("   Top Predictions:")
        for i, pred in enumerate(result['predictions'][:3]):
            print(f"   {i+1}. {pred['class']}: {pred['confidence']:.1f}%")
    
    # Create visualizations
    print("\n📈 Creating analysis visualizations...")
    recognizer.visualize_predictions(results)
    
    # Clean up sample videos
    print("\n🧹 Cleaning up sample videos...")
    for video in sample_videos:
        if os.path.exists(video):
            os.remove(video)
    
    print("\n✅ Action recognition analysis complete!")
    print("📁 Check 'action_recognition_analysis.png' for detailed visualizations")

if __name__ == "__main__":
    main()

# 🧠 What This Modern Project Demonstrates:
# ✅ Multiple state-of-the-art model architectures (R3D-18, MC3-18, R(2+1)D-18)
# ✅ Complete Kinetics-400 dataset support (400 action classes)
# ✅ Automatic device selection (CUDA, MPS, CPU)
# ✅ Robust error handling and validation
# ✅ Batch processing capabilities
# ✅ Comprehensive visualization and analytics
# ✅ Modern Python practices and clean code structure
