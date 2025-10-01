# Project 146: Modern Action Recognition in Videos
# Description: Advanced action recognition using state-of-the-art deep learning models
# Features: Multiple model support, batch processing, web UI, and comprehensive visualization

import torch
import torchvision
import torchvision.transforms as T
import cv2
import numpy as np
import os
import json
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ActionRecognitionModel:
    """Modern action recognition model wrapper with multiple architectures support"""
    
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
        """Load complete Kinetics-400 class labels"""
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
        """Load the specified model"""
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
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def load_video_frames(self, video_path, num_frames=16, size=(112, 112)):
        """Load and preprocess video frames"""
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
            st.error(f"Error loading video: {e}")
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
            st.error(f"Error during prediction: {e}")
            return []

class MockVideoDatabase:
    """Mock database for storing video metadata and results"""
    
    def __init__(self):
        self.videos = []
        self.results = []
    
    def add_video(self, filename, duration, fps, size):
        """Add video metadata to database"""
        video_id = len(self.videos)
        self.videos.append({
            'id': video_id,
            'filename': filename,
            'duration': duration,
            'fps': fps,
            'size': size,
            'timestamp': pd.Timestamp.now()
        })
        return video_id
    
    def add_result(self, video_id, predictions, model_name):
        """Add prediction results to database"""
        self.results.append({
            'video_id': video_id,
            'predictions': predictions,
            'model_name': model_name,
            'timestamp': pd.Timestamp.now()
        })
    
    def get_video_history(self):
        """Get all video processing history"""
        return pd.DataFrame(self.videos)
    
    def get_result_history(self):
        """Get all prediction results"""
        return pd.DataFrame(self.results)

def create_sample_videos():
    """Create sample videos for testing (mock implementation)"""
    st.info("Creating sample videos for demonstration...")
    
    # Create a simple colored video for testing
    sample_videos = []
    
    # Create different colored rectangles as "actions"
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
    
    return sample_videos

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Action Recognition in Videos",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Advanced Action Recognition in Videos")
    st.markdown("**State-of-the-art deep learning models for video action classification**")
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'database' not in st.session_state:
        st.session_state.database = MockVideoDatabase()
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["r3d_18", "mc3_18", "r2plus1d_18"],
        help="Choose the action recognition model architecture"
    )
    
    device = st.sidebar.selectbox(
        "Device",
        ["auto", "cpu", "cuda", "mps"],
        help="Select computation device"
    )
    
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            st.session_state.model = ActionRecognitionModel(model_name, device)
            if st.session_state.model.load_model():
                st.sidebar.success(f"✅ {model_name} loaded successfully!")
            else:
                st.sidebar.error("❌ Failed to load model")
    
    # Main content
    if st.session_state.model is None:
        st.warning("⚠️ Please load a model first using the sidebar.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎥 Video Analysis", "📊 Batch Processing", "📈 Analytics", "🗄️ Database"])
    
    with tab1:
        st.header("Single Video Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload a video file",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video file for action recognition"
            )
            
            if uploaded_file is not None:
                # Save uploaded file
                video_path = f"temp_{uploaded_file.name}"
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load video frames
                video_tensor, duration, fps = st.session_state.model.load_video_frames(video_path)
                
                if video_tensor is not None:
                    st.success(f"✅ Video loaded: {duration:.2f}s, {fps:.1f} FPS")
                    
                    # Make prediction
                    if st.button("🔍 Analyze Video", type="primary"):
                        with st.spinner("Analyzing video..."):
                            predictions = st.session_state.model.predict(video_tensor)
                            
                            if predictions:
                                # Display results
                                st.subheader("🎯 Top 5 Predictions")
                                
                                for i, pred in enumerate(predictions):
                                    col_pred, col_conf = st.columns([3, 1])
                                    with col_pred:
                                        st.write(f"**{i+1}.** {pred['class'].replace('_', ' ').title()}")
                                    with col_conf:
                                        st.metric("Confidence", f"{pred['confidence']:.1f}%")
                                
                                # Add to database
                                video_id = st.session_state.database.add_video(
                                    uploaded_file.name, duration, fps, video_tensor.shape
                                )
                                st.session_state.database.add_result(
                                    video_id, predictions, model_name
                                )
                                
                                # Clean up
                                os.remove(video_path)
                
        with col2:
            st.subheader("📋 Sample Videos")
            if st.button("Generate Sample Videos"):
                sample_videos = create_sample_videos()
                st.success(f"Created {len(sample_videos)} sample videos!")
                
                for video in sample_videos:
                    if st.button(f"Analyze {video}"):
                        video_tensor, duration, fps = st.session_state.model.load_video_frames(video)
                        if video_tensor is not None:
                            predictions = st.session_state.model.predict(video_tensor)
                            st.write("**Predictions:**")
                            for pred in predictions[:3]:
                                st.write(f"- {pred['class']}: {pred['confidence']:.1f}%")
    
    with tab2:
        st.header("Batch Video Processing")
        
        uploaded_files = st.file_uploader(
            "Upload multiple video files",
            type=['mp4', 'avi', 'mov', 'mkv'],
            accept_multiple_files=True,
            help="Upload multiple videos for batch processing"
        )
        
        if uploaded_files:
            progress_bar = st.progress(0)
            results_container = st.container()
            
            if st.button("🚀 Process All Videos", type="primary"):
                all_results = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Save and process video
                    video_path = f"temp_{uploaded_file.name}"
                    with open(video_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    video_tensor, duration, fps = st.session_state.model.load_video_frames(video_path)
                    
                    if video_tensor is not None:
                        predictions = st.session_state.model.predict(video_tensor)
                        
                        # Add to database
                        video_id = st.session_state.database.add_video(
                            uploaded_file.name, duration, fps, video_tensor.shape
                        )
                        st.session_state.database.add_result(
                            video_id, predictions, model_name
                        )
                        
                        all_results.append({
                            'filename': uploaded_file.name,
                            'duration': duration,
                            'predictions': predictions
                        })
                    
                    # Clean up
                    if os.path.exists(video_path):
                        os.remove(video_path)
                
                # Display results
                with results_container:
                    st.subheader("📊 Batch Processing Results")
                    
                    for result in all_results:
                        with st.expander(f"📹 {result['filename']}"):
                            st.write(f"**Duration:** {result['duration']:.2f}s")
                            st.write("**Top Predictions:**")
                            for pred in result['predictions'][:3]:
                                st.write(f"- {pred['class']}: {pred['confidence']:.1f}%")
    
    with tab3:
        st.header("📈 Analytics Dashboard")
        
        if len(st.session_state.database.videos) > 0:
            # Video statistics
            st.subheader("Video Statistics")
            video_df = st.session_state.database.get_video_history()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Videos", len(video_df))
            with col2:
                st.metric("Total Duration", f"{video_df['duration'].sum():.1f}s")
            with col3:
                st.metric("Average FPS", f"{video_df['fps'].mean():.1f}")
            
            # Confidence distribution
            if len(st.session_state.database.results) > 0:
                st.subheader("Prediction Confidence Distribution")
                
                all_confidences = []
                for result in st.session_state.database.results:
                    for pred in result['predictions']:
                        all_confidences.append(pred['confidence'])
                
                fig = px.histogram(
                    x=all_confidences,
                    nbins=20,
                    title="Distribution of Prediction Confidences",
                    labels={'x': 'Confidence (%)', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Top predicted classes
                st.subheader("Most Predicted Actions")
                all_classes = []
                for result in st.session_state.database.results:
                    for pred in result['predictions']:
                        all_classes.append(pred['class'])
                
                class_counts = pd.Series(all_classes).value_counts().head(10)
                fig = px.bar(
                    x=class_counts.values,
                    y=class_counts.index,
                    orientation='h',
                    title="Top 10 Most Predicted Actions"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No videos processed yet. Upload some videos to see analytics!")
    
    with tab4:
        st.header("🗄️ Database Management")
        
        if len(st.session_state.database.videos) > 0:
            st.subheader("Video History")
            video_df = st.session_state.database.get_video_history()
            st.dataframe(video_df, use_container_width=True)
            
            if st.button("📥 Export Data"):
                csv = video_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="video_history.csv",
                    mime="text/csv"
                )
        else:
            st.info("No data in database yet.")

if __name__ == "__main__":
    main()
