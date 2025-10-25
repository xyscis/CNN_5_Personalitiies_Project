# python -m venv venv
# .\venv\Scripts\activate

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
    batch_size = 16         # Process 8 videos at once
    epochs = 5             # Train for 20 rounds
    learning_rate = 0.001    # How fast to learn
    
    # Computer settings
    use_gpu = torch.cuda.is_available()
    num_workers = 4          # Parallel data loading


# ============================================
# ACCURACY METRIC
# ============================================
def calculate_accuracy(predictions, targets):
    """
    Calculate accuracy: % of predictions within 0.15 of true value
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    errors = np.abs(predictions - targets)
    accuracy = np.mean(errors <= 0.15) * 100
    return accuracy


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
    all_predictions = []
    all_targets = []
    
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
        
        # Store predictions for accuracy calculation
        all_predictions.append(predicted_scores.detach())
        all_targets.append(true_scores.detach())
    
    # Calculate metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    avg_loss = total_loss / count
    accuracy = calculate_accuracy(all_predictions, all_targets)
    
    return avg_loss, accuracy


# ============================================
# STEP 6: Evaluation Function
# ============================================
def test_model(model, data_loader, device):
    """Test the model without updating it"""
    model.eval()
    total_loss = 0
    count = 0
    all_predictions = []
    all_targets = []
    
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
            
            # Store predictions for accuracy calculation
            all_predictions.append(predicted_scores)
            all_targets.append(true_scores)
    
    # Calculate metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    avg_loss = total_loss / count
    accuracy = calculate_accuracy(all_predictions, all_targets)
    
    return avg_loss, accuracy


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
    # print(f"   Train: {len(train_labels)} videos")
    # print(f"   Val: {len(val_labels)} videos")
    # print(f"   Test: {len(test_labels)} videos")
    
    # # Create datasets
    # print("\n2. Creating datasets...")
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
        train_loss, train_acc = train_one_round(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_acc = test_model(model, val_loader, device)
        
        # Print progress
        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}, train_acc={train_acc:.1f}%, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.1f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            print(f"   → Best! (improved from {best_val_loss:.4f})")
            best_val_loss = val_loss
            best_model = {k: v.cpu() for k, v in model.state_dict().items()}
        
        # Reduce learning rate at milestones
        scheduler.step()
    
    # Load best model
    print("\n5. Loading best model...")
    if best_model:
        model.load_state_dict(best_model)
    
    # Final test
    test_loss, test_acc = test_model(model, test_loader, device)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.1f}%")
    print("\nWhat this means:")
    print("  • Loss: Lower is better (measures prediction error)")
    print("  • Accuracy: % of predictions within ±0.15 of true value")
    print("  • Scores range from 0.0 to 1.0")


if __name__ == "__main__":
    main()