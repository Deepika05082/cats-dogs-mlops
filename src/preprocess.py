import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

class CatsDogsDataset(Dataset):
    def __init__(self, root, transform=transform):
        self.root = root
        self.transform = transform
        self.files = [os.path.join(root, f) for f in os.listdir(root)]

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        label = 0 if "cat" in img_path.lower() else 1
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label
