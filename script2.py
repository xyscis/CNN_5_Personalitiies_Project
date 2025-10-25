# import pickle
# from pathlib import Path
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms

# # ============================================
# # Configuration
# # ============================================
# class Config:
#     """All hyperparameters and paths in one place"""
#     # Paths
#     root = "ChaLearn2016_tiny"
#     train_dir = "train"
#     val_dir = "valid"
#     test_dir = "test"
#     ann_dir = "annotation"
#     ann_train = "annotation_training.pkl"
#     ann_val = "annotation_validation.pkl"
#     ann_test = "annotation_test.pkl"
    
#     # Video hyperparameters
#     num_frames = 16          # Number of frames per video clip (as per diagram)
#     img_size = 112           # Frame size 112x112 (as per diagram)
    
#     # Training hyperparameters
#     batch_size = 8           # Can increase if you have memory
#     num_workers = 4
#     epochs = 30
#     learning_rate = 1e-3
#     weight_decay = 1e-4
#     seed = 42
    
#     # Training strategy
#     lr_decay_epochs = [15, 25]
#     lr_decay_factor = 0.1
    
#     # Device
#     device = "cuda" if torch.cuda.is_available() else "cpu"

# TRAITS = ["extraversion", "agreeableness", "conscientiousness", 
#           "neuroticism", "openness", "interview"]


# # ============================================
# # Data Loading
# # ============================================
# def load_annotations(pickle_path: Path) -> dict:
#     """Load annotation pickle file"""
#     with open(pickle_path, "rb") as f:
#         try:
#             data = pickle.load(f)
#         except:
#             f.seek(0)
#             data = pickle.load(f, encoding="latin1")
    
#     all_filenames = set()
#     for trait in TRAITS:
#         if trait in data and isinstance(data[trait], dict):
#             all_filenames.update(data[trait].keys())
    
#     annotations = {}
#     for filename in all_filenames:
#         trait_values = []
#         for trait in TRAITS:
#             if trait not in data or filename not in data[trait]:
#                 break
#             trait_values.append(float(data[trait][filename]))
#         else:
#             annotations[str(filename)] = np.array(trait_values, dtype=np.float32)
    
#     return annotations


# def extract_frames(video_path: Path, num_frames: int, frame_size: int) -> np.ndarray:
#     """Extract uniformly sampled frames from video"""
#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video: {video_path}")
    
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_indices = np.linspace(0, max(0, total_frames - 1), num=num_frames).astype(int)
    
#     frames = []
#     for idx in frame_indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
#         success, frame = cap.read()
#         if not success or frame is None:
#             break
        
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = cv2.resize(frame, (frame_size, frame_size))
#         frames.append(frame)
    
#     cap.release()
    
#     if not frames:
#         raise RuntimeError(f"No frames read from {video_path}")
    
#     while len(frames) < num_frames:
#         frames.append(frames[-1].copy())
    
#     return np.stack(frames[:num_frames], axis=0)


# # ============================================
# # Dataset
# # ============================================
# class PersonalityDataset(Dataset):
#     """Dataset for personality trait prediction from video clips"""
    
#     def __init__(self, video_dir: Path, annotations: dict, num_frames: int, frame_size: int):
#         self.num_frames = num_frames
#         self.frame_size = frame_size
#         self.annotations = {str(k): v.astype(np.float32) for k, v in annotations.items()}
        
#         # Video transform
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
        
#         ann_by_lowercase = {k.lower(): k for k in self.annotations.keys()}
#         ann_by_stem = {Path(k).stem.lower(): k for k in self.annotations.keys()}
        
#         video_extensions = ".mp4"
#         video_files = [p for p in video_dir.rglob("*") 
#                       if p.is_file() and p.suffix.lower() in video_extensions]
        
#         self.samples = []
#         for video_path in sorted(video_files):
#             filename = video_path.name
#             stem = video_path.stem
            
#             annotation_key = None
#             if filename in self.annotations:
#                 annotation_key = filename
#             elif filename.lower() in ann_by_lowercase:
#                 annotation_key = ann_by_lowercase[filename.lower()]
#             elif stem.lower() in ann_by_stem:
#                 annotation_key = ann_by_stem[stem.lower()]
            
#             if annotation_key:
#                 self.samples.append((video_path, annotation_key))
        
#         if not self.samples:
#             raise RuntimeError(f"No videos matched with annotations in {video_dir}")
        
#         print(f"Loaded {len(self.samples)} videos from {video_dir}")
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         video_path, annotation_key = self.samples[idx]
        
#         # Extract frames: [T, H, W, 3]
#         frames = extract_frames(video_path, self.num_frames, self.frame_size)
#         frames = torch.stack([self.transform(frame) for frame in frames], dim=0)
#         frames = frames.permute(1, 0, 2, 3)  # [3, T, H, W] for 3D Conv
        
#         target = torch.from_numpy(self.annotations[annotation_key])
        
#         return frames, target, video_path.name


# # ============================================
# # Model Architecture (Exact from Diagram)
# # ============================================
# class Visual3DCNN(nn.Module):
#     """
#     3D CNN architecture matching the diagram exactly:
#     Input: [3, 6, 112, 112] - Aligned Selected Face Images
    
#     Block 1: 3D Conv (3x5x5, stride 1x1x1) + ReLU + MaxPool (2x2x2, stride 2x2x2)
#     → Output: [16, 2, 54, 54]
    
#     Block 2: 3D Conv (2x5x5, stride 1x1x1) + ReLU + MaxPool (1x2x2, stride 1x2x2)
#     → Output: [16, 1, 25, 25]
    
#     Block 3: 3D Conv (1x5x5, stride 1x1x1) + ReLU
#     → Output: [1, 2, 24, 24]

#     Flatten → FC (1152)
#     """
    
#     def __init__(self, num_classes=6):
#         super(Visual3DCNN, self).__init__()
        
#         # Block 1: [3, 6, 112, 112] -> [16, 2, 54, 54]
#         self.conv1 = nn.Conv3d(
#             in_channels=3,
#             out_channels=16,
#             kernel_size=(3, 5, 5),
#             stride=(1, 1, 1),
#             padding=(1, 2, 2)  # Same padding to maintain spatial dims before pooling
#         )
#         self.relu1 = nn.ReLU(inplace=True)
#         self.pool1 = nn.MaxPool3d(
#             kernel_size=(2, 2, 2),
#             stride=(2, 2, 2)
#         )
        
#         # Block 2: [16, 2, 54, 54] -> [16, 1, 25, 25]
#         self.conv2 = nn.Conv3d(
#             in_channels=16,
#             out_channels=16,
#             kernel_size=(2, 5, 5),
#             stride=(1, 1, 1),
#             padding=(0, 2, 2)  # No temporal padding since kernel is 2
#         )
#         self.relu2 = nn.ReLU(inplace=True)
#         self.pool2 = nn.MaxPool3d(
#             kernel_size=(1, 2, 2),
#             stride=(1, 2, 2)
#         )
        
#         # Block 3: [16, 1, 25, 25] -> [1, 1, 21, 21]
#         self.conv3 = nn.Conv3d(
#             in_channels=16,
#             out_channels=1,
#             kernel_size=(1, 5, 5),
#             stride=(1, 1, 1),
#             padding=(0, 0, 0)  # No padding - reduces spatial dims
#         )
#         self.relu3 = nn.ReLU(inplace=True)
        
#     # After conv3: [batch, 1, 2, 24, 24]
#     # Flatten to: [batch, 1*2*24*24] = [batch, 1152]
        
#     # Fully connected layer (match flattened conv output)
#         self.fc = nn.Linear(1152, num_classes)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         """
#         Args:
#             x: [batch, 3, 6, 112, 112] - video clip
#         Returns:
#             [batch, num_classes] - personality trait predictions
#         """
#         # Block 1
#         x = self.conv1(x)      # [B, 16, 6, 112, 112]
#         x = self.relu1(x)
#         x = self.pool1(x)      # [B, 16, 3, 56, 56]
        
#         # Block 2
#         x = self.conv2(x)      # [B, 16, 2, 56, 56]
#         x = self.relu2(x)
#         x = self.pool2(x)      # [B, 16, 2, 28, 28]
        
#     # Block 3
#         x = self.conv3(x)      # [B, 1, 2, 24, 24]
#         x = self.relu3(x)
        
#         # Flatten
#         x = x.view(x.size(0), -1)  # [B, 1*2*24*24] = [B, 1152]
        
#         # Fully connected
#         x = self.fc(x)         # [B, num_classes]
#         return self.sigmoid(x)


# # ============================================
# # Training Functions
# # ============================================
# def train_one_epoch(model, dataloader, optimizer, device):
#     """Train for one epoch"""
#     model.train()
#     total_loss = 0.0
#     total_correct = 0
#     total_predictions = 0
#     num_samples = 0
#     loss_fn = nn.MSELoss()
#     threshold = 0.1  # Consider prediction correct if within 0.1 of target
    
#     for frames, targets, _ in dataloader:
#         frames = frames.to(device)
#         targets = targets.to(device)
        
#         optimizer.zero_grad()
#         predictions = model(frames)
#         loss = loss_fn(predictions, targets)
#         loss.backward()
#         optimizer.step()
        
#         # Calculate accuracy (predictions within threshold of target)
#         correct = (torch.abs(predictions - targets) < threshold).float()
#         total_correct += correct.sum().item()
#         total_predictions += predictions.numel()
        
#         total_loss += loss.item() * frames.size(0)
#         num_samples += frames.size(0)
    
#     avg_loss = total_loss / num_samples
#     accuracy = 100.0 * total_correct / total_predictions
#     return avg_loss, accuracy


# def evaluate(model, dataloader, device):
#     """Evaluate model"""
#     model.eval()
#     total_loss = 0.0
#     total_correct = 0
#     total_predictions = 0
#     num_samples = 0
#     loss_fn = nn.MSELoss()
#     threshold = 0.1  # Consider prediction correct if within 0.1 of target
    
#     with torch.no_grad():
#         for frames, targets, _ in dataloader:
#             frames = frames.to(device)
#             targets = targets.to(device)
            
#             predictions = model(frames)
#             loss = loss_fn(predictions, targets)
            
#             # Calculate accuracy (predictions within threshold of target)
#             correct = (torch.abs(predictions - targets) < threshold).float()
#             total_correct += correct.sum().item()
#             total_predictions += predictions.numel()
            
#             total_loss += loss.item() * frames.size(0)
#             num_samples += frames.size(0)
    
#     avg_loss = total_loss / num_samples
#     accuracy = 100.0 * total_correct / total_predictions
#     return avg_loss, accuracy


# # ============================================
# # Main Training Loop
# # ============================================
# def main():
#     torch.manual_seed(Config.seed)
#     np.random.seed(Config.seed)
    
#     root = Path(Config.root)
    
#     # Load annotations
#     print("Loading annotations...")
#     train_annotations = load_annotations(root / Config.ann_dir / Config.ann_train)
#     val_annotations = load_annotations(root / Config.ann_dir / Config.ann_val)
#     test_annotations = load_annotations(root / Config.ann_dir / Config.ann_test)
    
#     # Create datasets
#     print("Creating datasets...")
#     train_dataset = PersonalityDataset(
#         root / Config.train_dir, train_annotations, Config.num_frames, Config.img_size)
#     val_dataset = PersonalityDataset(
#         root / Config.val_dir, val_annotations, Config.num_frames, Config.img_size)
#     test_dataset = PersonalityDataset(
#         root / Config.test_dir, test_annotations, Config.num_frames, Config.img_size)
    
#     # Create dataloaders
#     train_loader = DataLoader(
#         train_dataset, batch_size=Config.batch_size, shuffle=True,
#         num_workers=Config.num_workers, pin_memory=True)
#     val_loader = DataLoader(
#         val_dataset, batch_size=Config.batch_size, shuffle=False,
#         num_workers=Config.num_workers, pin_memory=True)
#     test_loader = DataLoader(
#         test_dataset, batch_size=Config.batch_size, shuffle=False,
#         num_workers=Config.num_workers, pin_memory=True)
    
#     # Create model matching the diagram
#     print(f"Creating Visual 3D CNN (from diagram) on {Config.device}...")
#     model = Visual3DCNN(num_classes=6).to(Config.device)
    
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total parameters: {total_params:,}")
#     print("\nArchitecture (matching diagram):")
#     print("  Input: [3, 6, 112, 112]")
#     print("  Conv1: 3D Conv (3x5x5) → ReLU → MaxPool (2x2x2) → [16, 2, 54, 54]")
#     print("  Conv2: 3D Conv (2x5x5) → ReLU → MaxPool (1x2x2) → [16, 1, 25, 25]")
#     print("  Conv3: 3D Conv (1x5x5) → ReLU → [1, 1, 21, 21]")
#     print("  Conv3: 3D Conv (1x5x5) → ReLU → [1, 2, 24, 24]")
#     print("  Flatten: 1*2*24*24 = 1152")
#     print("  FC: 1152 → 6 (personality traits)")
    
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=Config.learning_rate,
#         weight_decay=Config.weight_decay
#     )
    
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(
#         optimizer, 
#         milestones=Config.lr_decay_epochs, 
#         gamma=Config.lr_decay_factor
#     )
    
#     # Training loop
#     print(f"\nTraining Visual 3D CNN for {Config.epochs} epochs...")
#     best_val_loss = float('inf')
#     best_model_state = None
    
#     for epoch in range(1, Config.epochs + 1):
#         train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, Config.device)
#         val_loss, val_acc = evaluate(model, val_loader, Config.device)
        
#         current_lr = optimizer.param_groups[0]['lr']
#         print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")
        
#         if val_loss < best_val_loss:
#             print(f"  ✓ New best! (val_loss: {best_val_loss:.4f} → {val_loss:.4f})")
#             best_val_loss = val_loss
#             best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
#         scheduler.step()
    
#     # Load best model
#     if best_model_state:
#         print("\nLoading best model for testing...")
#         model.load_state_dict(best_model_state)
    
#     # Final test
#     test_loss, test_acc = evaluate(model, test_loader, Config.device)
#     print(f"\n{'='*60}")
#     print(f"FINAL TEST RESULTS")
#     print(f"{'='*60}")
#     print(f"Test Loss: {test_loss:.4f}")
#     print(f"Test Accuracy: {test_acc:.2f}%")
#     print(f"\nThis architecture exactly matches your diagram:")
#     print(f"  • 3 3D Conv layers with specific kernel sizes")
#     print(f"  • Progressive dimensionality reduction")
#     print(f"  • Final flatten to 1152 features → 6 traits")
#     print(f"\nNote: Accuracy is % of predictions within 0.1 threshold of target value")


# if __name__ == "__main__":
#     main()


import pickle
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ============================================
# SETTINGS - Change these if needed
# ============================================
class Settings:
    # Where your data is
    data_folder = "ChaLearn2016_tiny"
    
    # How to train
    frames_per_video = 16    # Sample 16 frames from each video
    frame_size = 112         # Resize to 112x112 pixels
    batch_size = 8           # Process 8 videos at once
    epochs = 30              # Train for 30 rounds
    learning_rate = 0.001    # How fast to learn
    
    # Computer settings
    use_gpu = torch.cuda.is_available()
    num_workers = 4          # Parallel data loading


# ============================================
# STEP 1: Load Labels (Ground Truth)
# ============================================
def load_labels(pickle_file):
    """
    Load the labels for each video.
    Returns: dictionary like {'video.mp4': [0.5, 0.7, 0.6, 0.4, 0.8, 0.5]}
    These 6 numbers are: extraversion, agreeableness, conscientiousness, 
                         neuroticism, openness, interview score
    """
    with open(pickle_file, "rb") as f:
        try:
            data = pickle.load(f)
        except:
            f.seek(0)
            data = pickle.load(f, encoding="latin1")
    
    traits = ["extraversion", "agreeableness", "conscientiousness", 
              "neuroticism", "openness", "interview"]
    
    # Find all video filenames
    all_files = set()
    for trait in traits:
        if trait in data and isinstance(data[trait], dict):
            all_files.update(data[trait].keys())
    
    # Build label dictionary
    labels = {}
    for filename in all_files:
        scores = []
        has_all_traits = True
        
        for trait in traits:
            if trait in data and filename in data[trait]:
                scores.append(float(data[trait][filename]))
            else:
                has_all_traits = False
                break
        
        if has_all_traits:
            labels[filename] = np.array(scores, dtype=np.float32)
    
    return labels


# ============================================
# STEP 2: Extract Frames from Video
# ============================================
def get_frames_from_video(video_path, num_frames, size):
    """
    Extract evenly spaced frames from a video.
    If video has 100 frames and we want 16, we take frames: 0, 6, 13, 19, ...
    Returns: numpy array [num_frames, size, size, 3]
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(0, total - 1), num=num_frames).astype(int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if not success:
            break
        
        # Convert BGR to RGB and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        frames.append(frame)
    
    cap.release()
    
    if not frames:
        raise RuntimeError(f"No frames from: {video_path}")
    
    # Pad with last frame if video is too short
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    
    return np.stack(frames[:num_frames], axis=0)


# ============================================
# STEP 3: Dataset Class (Connects Videos to Labels)
# ============================================
class VideoDataset(Dataset):
    """
    PyTorch Dataset that:
    1. Finds all videos in a folder
    2. Matches them with labels
    3. Returns (video_frames, personality_scores)
    """
    
    def __init__(self, video_folder, labels_dict, num_frames, frame_size):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.labels = labels_dict
        
        # Normalize frames (standard for pretrained models)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
        
        # Find all .mp4 files
        video_files = list(video_folder.glob("**/*.mp4"))
        
        # Match videos with labels
        self.video_label_pairs = []
        for video_path in video_files:
            video_name = video_path.name
            
            # Try exact match or lowercase match
            label_key = None
            if video_name in self.labels:
                label_key = video_name
            elif video_name.lower() in {k.lower(): k for k in self.labels}:
                label_key = {k.lower(): k for k in self.labels}[video_name.lower()]
            
            if label_key:
                self.video_label_pairs.append((video_path, label_key))
        
        if not self.video_label_pairs:
            raise RuntimeError(f"No videos found in {video_folder}")
        
        print(f"Found {len(self.video_label_pairs)} videos")
    
    def __len__(self):
        return len(self.video_label_pairs)
    
    def __getitem__(self, idx):
        video_path, label_key = self.video_label_pairs[idx]
        
        # Get frames: [16, 112, 112, 3]
        frames = get_frames_from_video(video_path, self.num_frames, self.frame_size)
        
        # Normalize each frame and stack: [16, 3, 112, 112]
        frames = torch.stack([self.normalize(f) for f in frames], dim=0)
        
        # Rearrange to [3, 16, 112, 112] for 3D CNN
        frames = frames.permute(1, 0, 2, 3)
        
        # Get personality scores
        scores = torch.from_numpy(self.labels[label_key])
        
        return frames, scores


# ============================================
# STEP 4: The 3D CNN Model
# ============================================
class PersonalityModel(nn.Module):
    """
    3D CNN that learns from video clips.
    
    Architecture:
    1. Input: [3, 16, 112, 112] = 16 RGB frames of 112x112
    2. 3D Conv Block 1 → reduces to [16, 8, 56, 56]
    3. 3D Conv Block 2 → reduces to [16, 7, 28, 28]
    4. 3D Conv Block 3 → reduces to [1, 7, 24, 24]
    5. Flatten → [1008 features]
    6. Linear → [6 personality scores]
    """
    
    def __init__(self):
        super().__init__()
        
        # Block 1: Extract low-level features
        self.block1 = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        # Block 2: Extract mid-level features
        self.block2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(2, 5, 5), padding=(0, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        # Block 3: Extract high-level features
        self.block3 = nn.Sequential(
            nn.Conv3d(16, 1, kernel_size=(1, 5, 5), padding=(0, 0, 0)),
            nn.ReLU()
        )
        
        # Calculate how many features we get after convolutions
        self.feature_size = self._calculate_feature_size()
        
        # Final prediction layer
        self.predict = nn.Linear(self.feature_size, 6)
        self.sigmoid = nn.Sigmoid()  # Output between 0 and 1
    
    def _calculate_feature_size(self):
        """Figure out the size after all convolutions"""
        dummy = torch.zeros(1, 3, Settings.frames_per_video, 
                           Settings.frame_size, Settings.frame_size)
        x = self.block1(dummy)
        x = self.block2(x)
        x = self.block3(x)
        return x.view(1, -1).size(1)
    
    def forward(self, video):
        """
        video: [batch, 3, 16, 112, 112]
        returns: [batch, 6] personality scores
        """
        x = self.block1(video)
        x = self.block2(x)
        x = self.block3(x)
        
        # Flatten to 1D
        x = x.view(x.size(0), -1)
        
        # Predict 6 scores
        x = self.predict(x)
        return self.sigmoid(x)


# ============================================
# STEP 5: Training Function
# ============================================
def train_one_round(model, data_loader, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    count = 0
    
    for videos, true_scores in data_loader:
        videos = videos.to(device)
        true_scores = true_scores.to(device)
        
        # Predict
        predicted_scores = model(videos)
        
        # Calculate error (Mean Squared Error)
        loss = nn.MSELoss()(predicted_scores, true_scores)
        
        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * videos.size(0)
        count += videos.size(0)
    
    return total_loss / count


# ============================================
# STEP 6: Evaluation Function
# ============================================
def test_model(model, data_loader, device):
    """Test the model without updating it"""
    model.eval()
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for videos, true_scores in data_loader:
            videos = videos.to(device)
            true_scores = true_scores.to(device)
            
            # Predict
            predicted_scores = model(videos)
            
            # Calculate error
            loss = nn.MSELoss()(predicted_scores, true_scores)
            
            total_loss += loss.item() * videos.size(0)
            count += videos.size(0)
    
    return total_loss / count


# ============================================
# STEP 7: Main Training Script
# ============================================
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = "cuda" if Settings.use_gpu else "cpu"
    print(f"Using device: {device}")
    
    # Create folder paths
    root = Path(Settings.data_folder)
    ann_dir = root / "annotation"
    
    # Load labels
    print("\n1. Loading labels...")
    train_labels = load_labels(ann_dir / "annotation_training.pkl")
    val_labels = load_labels(ann_dir / "annotation_validation.pkl")
    test_labels = load_labels(ann_dir / "annotation_test.pkl")
    print(f"   Train: {len(train_labels)} videos")
    print(f"   Val: {len(val_labels)} videos")
    print(f"   Test: {len(test_labels)} videos")
    
    # Create datasets
    print("\n2. Creating datasets...")
    train_data = VideoDataset(root / "train", train_labels, 
                              Settings.frames_per_video, Settings.frame_size)
    val_data = VideoDataset(root / "valid", val_labels,
                           Settings.frames_per_video, Settings.frame_size)
    test_data = VideoDataset(root / "test", test_labels,
                            Settings.frames_per_video, Settings.frame_size)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=Settings.batch_size, 
                             shuffle=True, num_workers=Settings.num_workers)
    val_loader = DataLoader(val_data, batch_size=Settings.batch_size,
                           shuffle=False, num_workers=Settings.num_workers)
    test_loader = DataLoader(test_data, batch_size=Settings.batch_size,
                            shuffle=False, num_workers=Settings.num_workers)
    
    # Create model
    print("\n3. Creating 3D CNN model...")
    model = PersonalityModel().to(device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Feature size after convolutions: {model.feature_size}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Settings.learning_rate)
    
    # Learning rate scheduler (reduce LR at epochs 15 and 25)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones=[15, 25], 
                                                     gamma=0.1)
    
    # Training loop
    print(f"\n4. Training for {Settings.epochs} epochs...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(1, Settings.epochs + 1):
        # Train
        train_loss = train_one_round(model, train_loader, optimizer, device)
        
        # Validate
        val_loss = test_model(model, val_loader, device)
        
        # Print progress
        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}, "
              f"val_loss={val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            print(f"   → Best so far! (improved from {best_val_loss:.4f})")
            best_val_loss = val_loss
            best_model = {k: v.cpu() for k, v in model.state_dict().items()}
        
        # Reduce learning rate at milestones
        scheduler.step()
    
    # Load best model
    print("\n5. Loading best model...")
    if best_model:
        model.load_state_dict(best_model)
    
    # Final test
    test_loss = test_model(model, test_loader, device)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Test Loss: {test_loss:.4f}")
    print("\nWhat this model does:")
    print("  • Takes 16 frames from each video")
    print("  • Uses 3D convolutions to learn motion patterns")
    print("  • Predicts 6 personality trait scores (0-1 range)")
    print("\nLower loss = better predictions")


if __name__ == "__main__":
    main()