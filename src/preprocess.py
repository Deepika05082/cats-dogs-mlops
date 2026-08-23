import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

class CatsDogsDataset(Dataset):
    def __init__(self, root, transform=transform):
        self.root = root
        self.transform = transform
        self.files = []
        # Walk into Cat and Dog folders
        for label_name in ["Cat", "Dog"]:
            folder = os.path.join(root, label_name)
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                if os.path.isfile(fpath):   # only add actual files
                    self.files.append((fpath, 0 if label_name=="Cat" else 1))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label
