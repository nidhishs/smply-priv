import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import Compose, Resize, Normalize
from transformers import VideoMAEModel
from torchvision.io import read_video

class VideoDataset(Dataset):
    """Custom Dataset for loading videos and their labels."""
    
    def __init__(self, video_dir, label_file, transform=None):
        self.video_dir = video_dir
        self.transform = transform
        self.video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
        
        self.labels = self.load_labels(label_file)
        self.num_classes = len(set(self.labels.values()))
        print(f"Number of classes: {self.num_classes}")
        
    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video = self.load_video(video_path)
        if self.transform:
            video = self.transform(video)
        
        label = self.labels[os.path.basename(video_path)]
        return video, label

    def load_video(self, path):
        video, _, _ = read_video(path, pts_unit='sec')
        video = video.permute(3, 0, 1, 2)  # Change to C, T, H, W format
        return video

    def load_labels(self, label_file):
        labels = {}
        with open(label_file, 'r') as f:
            for line in f:
                video_name, label = line.strip().split(',')
                labels[video_name] = int(label)
        return labels
      

def downstream_eval(args):
    transform = Compose([
        Resize((224, 224)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = VideoDataset(args.data_dir, transform=transform, label_file=args.train_label_file)
    test_dataset = VideoDataset(args.data_dir, transform=transform, label_file=args.test_label_file)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = VideoMAEModel.from_pretrained(args.model_dir)
    if args.eval_type == 'linear_update':
        for param in model.parameters():
            param.requires_grad = False
    
    class VideoMAEWithClassifier(torch.nn.Module):
        def __init__(self, model, num_classes):
            super(VideoMAEWithClassifier, self).__init__()
            self.model = model
            self.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
        
        def forward(self, videos):
            outputs = self.model(videos)
            hidden_states = outputs.last_hidden_state[:, 0]
            logits = self.classifier(hidden_states)
            return logits

    model = VideoMAEWithClassifier(model, train_dataset.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs - args.warmup_epochs)

    for epoch in range(args.num_epochs):
        model.train()
        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss.item()}")
        
        if epoch >= args.warmup_epochs:
            scheduler.step()
        
        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for videos, labels in test_loader:
                    videos, labels = videos.to(device), labels.to(device)
                    outputs = model(videos)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = correct / total
            print(f"Test Accuracy after epoch {epoch+1}: {accuracy}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Downstream Evaluation for VideoMAE")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--model_dir', type=str, required=True, help="Path to the pre-trained model directory")
    parser.add_argument('--train_label_file', type=str, required=True, help="Path to the training label file")
    parser.add_argument('--test_label_file', type=str, required=True, help="Path to the testing label file")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for evaluation")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument('--num_epochs', type=int, default=30, help="Number of training epochs")
    parser.add_argument('--warmup_epochs', type=int, default=5, help="Number of warmup epochs")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of data loading workers")
    parser.add_argument('--eval_interval', type=int, default=5, help="Interval of epochs to evaluate the model")
    parser.add_argument('--eval_type', type=str, required=True, choices=['linear_update', 'fine_tune'], help="Evaluation type: 'linear_update' or 'fine_tune'")
    args = parser.parse_args()
    downstream_eval(args)
