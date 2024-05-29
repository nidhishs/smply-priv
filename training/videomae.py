import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import Compose, Resize, Normalize
import imageio
from transformers import VideoMAEForPreTraining, VideoMAEConfig, VideoMAEModel

class VideoDataset(torch.utils.data.Dataset):
    """Custom Dataset for loading videos."""
    
    def __init__(self, data_dir, transform=None, label_file=None):
        self.data_dir = data_dir
        self.transform = transform
        self.video_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.mp4')]
            
        self.labels = None
        if label_file:
            print(f"Loading labels from {label_file}")
            self.labels = self.load_labels(label_file)
        
        self.num_classes = len(set(self.labels.values())) if self.labels else 0
        print(f"Number of classes: {self.num_classes}")
        
    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video = self.load_video(video_path)
        if self.transform:
            video = self.transform(video)
        
        label = self.labels[os.path.basename(video_path)] if self.labels else 0
        return video, label

    def load_video(self, path):
        try:
            # Load video frames
            reader = imageio.get_reader(path, 'ffmpeg')
            frames = [torch.tensor(frame).permute(2, 0, 1) for frame in reader]  # Change (H, W, C) to (C, H, W)
            reader.close()

            # Convert to tensor and normalize
            video_tensor = torch.stack(frames).float() / 255.0
            
            # Resize frames
            resize_transform = Resize((224, 224))
            video_tensor = torch.stack([resize_transform(frame) for frame in video_tensor])
            
            # Select a fixed number of frames (T = 16), resize if necessary
            T = 16
            if video_tensor.size(0) > T:
                start_frame = torch.randint(0, video_tensor.size(0) - T + 1, (1,)).item()
                video_tensor = video_tensor[start_frame:start_frame + T]
            elif video_tensor.size(0) < T:
                # If less than T frames, pad with zeros
                padding = torch.zeros((T - video_tensor.size(0), 3, 224, 224))
                video_tensor = torch.cat((video_tensor, padding), dim=0)
            
            return video_tensor
        except Exception as e:
            print(f"Error loading video {path}: {e}")
            return torch.zeros((16, 3, 224, 224))  # Return an empty tensor if there's an error

    def load_labels(self, label_file):
        labels = {}
        print(f"Reading labels from {label_file}")
        with open(label_file, 'r') as f:
            for line in f:
                video_name, label = line.strip().split(',')
                labels[video_name] = int(label)
        return labels

def create_bool_masked_pos(batch_size, sequence_length, mask_ratio=0.9):
    """Create a boolean mask for masking patches."""
    num_masked = int(sequence_length * mask_ratio)
    bool_masked_pos = torch.zeros((batch_size, sequence_length), dtype=torch.bool)
    for i in range(batch_size):
        mask_indices = torch.randperm(sequence_length)[:num_masked]
        bool_masked_pos[i, mask_indices] = True
    return bool_masked_pos

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VideoMAE Pre-training and Supervised Alignment Script")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the custom dataset directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the trained model")
    parser.add_argument('--label_file', type=str, required=False, help="Path to the label file")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=8e-4, help="Initial learning rate")
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--warmup_epochs', type=int, default=10, help="Number of warmup epochs")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of data loading workers")
    parser.add_argument('--stage', type=str, required=True, choices=['pretrain', 'align'], help="Training stage: 'pretrain' or 'align'")
    args = parser.parse_args()
    return args

def pretrain(args):
    # Load dataset
    transform = Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = VideoDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Load VideoMAE model
    config = VideoMAEConfig()
    model = VideoMAEForPreTraining(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs - args.warmup_epochs)

    # Training loop
    model.train()
    for epoch in range(args.num_epochs):
        for batch in dataloader:
            videos, _ = batch
            videos = videos.to(device)
            optimizer.zero_grad()
            
            # Create bool_masked_pos tensor
            batch_size, num_frames, _, _, _ = videos.shape
            sequence_length = (num_frames // model.config.tubelet_size) * (model.config.image_size // model.config.patch_size) ** 2
            bool_masked_pos = create_bool_masked_pos(batch_size, sequence_length).to(device)
            
            # Forward pass
            outputs = model(videos, bool_masked_pos=bool_masked_pos)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss.item()}")

        # Adjust learning rate
        if epoch >= args.warmup_epochs:
            scheduler.step()

        # Save model checkpoint
        if (epoch + 1) % 10 == 0 or epoch == args.num_epochs - 1:
            model.save_pretrained(os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}"))

    # Save pretrained model
    model.save_pretrained(args.output_dir)
    print(f"Pre-training completed. Model saved to {args.output_dir}")

def align(args):
    # Load dataset
    transform = Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = VideoDataset(args.data_dir, transform=transform, label_file=args.label_file)
    print(f"Loaded dataset with {len(dataset)} samples and {dataset.num_classes} classes")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Load pre-trained encoder
    model = VideoMAEModel.from_pretrained(args.output_dir)
    
    # Add a linear classification layer
    class VideoMAEWithClassifier(torch.nn.Module):
        def __init__(self, model, num_classes):
            super(VideoMAEWithClassifier, self).__init__()
            self.model = model
            self.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
        
        def forward(self, videos):
            outputs = self.model(videos)
            hidden_states = outputs.last_hidden_state[:, 0]  # Use the [CLS] token hidden state
            logits = self.classifier(hidden_states)
            return logits
    
    model = VideoMAEWithClassifier(model, dataset.num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs - args.warmup_epochs)

    # Training loop
    model.train()
    for epoch in range(args.num_epochs):
        for videos, labels in dataloader:
            videos, labels = videos.to(device), labels.to(device)

            assert torch.all(labels >= 0) and torch.all(labels < dataset.num_classes), f"Labels are out of range: {labels}, num_classes: {dataset.num_classes}"

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(videos)
            loss = F.cross_entropy(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss.item()}")

        # Adjust learning rate
        if epoch >= args.warmup_epochs:
            scheduler.step()

        # Save model checkpoint
        if (epoch + 1) % 10 == 0 or epoch == args.num_epochs - 1:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))
    print(f"Supervised alignment completed. Model saved to {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    if args.stage == 'pretrain':
        pretrain(args)
    elif args.stage == 'align':
        align(args)
