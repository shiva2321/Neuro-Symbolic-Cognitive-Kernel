
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import cv2
import numpy as np

class CharacterDataset10x10(Dataset):
    def __init__(self, split="train", num_samples=None):
        print(f">> [CHAR_DATA] Loading MNIST via Torchvision ({split})...")
        
        # 1. Define Transforms: Resize to 10x10 immediately
        # MNIST is 28x28 PIL Images.
        self.transform = transforms.Compose([
            transforms.Resize((10, 10)),
            transforms.ToTensor(), # Converts to [0,1] float tensor [1, 10, 10]
        ])
        
        is_train = (split == "train")
        try:
            self.ds = datasets.MNIST(root='./data_mnist', train=is_train, download=True, transform=self.transform)
        except Exception as e:
            print(f"[ERROR] Failed to load Torchvision MNIST: {e}")
            self.ds = []

        # Subset if requested
        if num_samples and len(self.ds) > 0:
            indices = range(min(num_samples, len(self.ds)))
            self.ds = Subset(self.ds, indices)
            
        print(f">> [CHAR_DATA] Loaded {len(self.ds)} samples.")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Item is (tensor[1, 10, 10], label_int)
        img_tensor, label = self.ds[idx]
        
        # Temporal Expansion for SNN: [4, 10, 10]
        # We perform simple replication of the static frame
        frames = img_tensor.repeat(4, 1, 1) # [4, 10, 10]
        
        return frames, torch.tensor(label, dtype=torch.long)

class CharacterDatasetAlphanumeric(Dataset):
    """
    EMNIST Dataset (ByClass): 62 classes (0-9, A-Z, a-z)
    """
    def __init__(self, split="train", num_samples=None):
        print(f">> [CHAR_DATA] Loading EMNIST (ByClass) via Torchvision ({split})...")
        
        self.transform = transforms.Compose([
            transforms.Resize((10, 10)),
            transforms.ToTensor(),
        ])
        
        is_train = (split == "train")
        try:
            # EMNIST byclass has 62 classes. 
            # Note: EMNIST is transposed by default in some versions, but ToTensor + Resize handles most cases.
            self.ds = datasets.EMNIST(root='./data_mnist', split='byclass', train=is_train, download=True, transform=self.transform)
        except Exception as e:
            print(f"[ERROR] Failed to load EMNIST: {e}")
            self.ds = []

        if num_samples and len(self.ds) > 0:
            indices = np.random.choice(len(self.ds), min(num_samples, len(self.ds)), replace=False)
            self.ds = torch.utils.data.Subset(self.ds, indices)
            
        print(f">> [CHAR_DATA] Loaded {len(self.ds)} EMNIST samples.")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img_tensor, label = self.ds[idx]
        # EMNIST labels for byclass:
        # 0-9: 0-9
        # 10-35: A-Z
        # 36-61: a-z
        # This matches our model's head_chars [62] exactly.
        frames = img_tensor.repeat(4, 1, 1)
        return frames, torch.tensor(label, dtype=torch.long)

class TextToImageDataset(Dataset):
    def __init__(self, text, num_repeats=100):
        self.text = text
        self.num_repeats = num_repeats
        self.chars = list(text)
        
        # Alphanumeric Map: 0-9 (0-9), A-Z (10-35), a-z (36-61)
        self.map = {}
        for i in range(10): self.map[str(i)] = i
        for i in range(26): self.map[chr(ord('A')+i)] = i + 10
        for i in range(26): self.map[chr(ord('a')+i)] = i + 36

    def __len__(self):
        return len(self.chars) * self.num_repeats

    def __getitem__(self, idx):
        char = self.chars[idx % len(self.chars)]
        label = self.map.get(char, 0) # Fallback to 0 if unknown
        
        # Render character to 10x10
        img = np.zeros((10, 10), dtype=np.uint8)
        # Using a simple 1-pixel font or just CV2
        cv2.putText(img, char, (1, 9), cv2.FONT_HERSHEY_PLAIN, 0.7, 255, 1)
        
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0) # [1, 10, 10]
        
        # Temporal Expansion for SNN
        frames = img_tensor.repeat(4, 1, 1)
        
        return frames, torch.tensor(label, dtype=torch.long)

def get_dataloader(batch_size=32, split="train", num_samples=None, num_workers=4, mode="mnist"):
    if mode == "mnist":
        dataset = CharacterDataset10x10(split=split, num_samples=num_samples)
    else: # alphanumeric / emnist
        dataset = CharacterDatasetAlphanumeric(split=split, num_samples=num_samples)
        
    if len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=num_workers, pin_memory=True)

def get_text_dataloader(text, batch_size=32, num_repeats=100):
    ds = TextToImageDataset(text, num_repeats=num_repeats)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    # Test
    dl = get_text_dataloader("ABC123abc", batch_size=5)
    for x, y in dl:
        print(f"Text Batch X: {x.shape}, Y: {y.shape}")
        print(f"Sample Y: {y}")
        break
