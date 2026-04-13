import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MissingFramesDataset(Dataset):
    """
    Dataset for missing frames that need to be added to existing model
    """
    def __init__(self, frames_directory, imagesize=(224, 224)):
        self.frames_directory = frames_directory
        self.image_paths = []
        
        # Collect all image files
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            self.image_paths.extend(list(Path(frames_directory).rglob(ext)))
        
        self.transform = transforms.Compose([
            transforms.Resize(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.601, 0.601, 0.601], 
                               std=[0.340, 0.340, 0.340]),
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        return {
            "image": self.transform(image),
            "image_path": str(image_path),
            "image_name": os.path.basename(image_path)
        }

def create_missing_frames_loader(frames_dir, batch_size=1):
    """
    Create DataLoader for missing frames
    """
    dataset = MissingFramesDataset(frames_dir)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    return loader