import torch
import torch.nn as nn
import torch.optim as optim
from src.model import SimpleCNN
from src.preprocess import CatsDogsDataset
import mlflow
from torch.utils.data import DataLoader, random_split
from torch.utils.data import DataLoader, random_split

full_ds = CatsDogsDataset("data/PetImages")
train_size = int(0.8 * len(full_ds))
val_size   = len(full_ds) - train_size

train_ds, val_ds = random_split(full_ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)


model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

mlflow.start_run()
for epoch in range(5):
    model.train()
    for x,y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    # validation
    model.eval()
    correct, total = 0,0
    with torch.no_grad():
        for x,y in val_loader:
            preds = model(x).argmax(1)
            correct += (preds==y).sum().item()
            total += y.size(0)
    acc = correct/total
    mlflow.log_metric("val_acc", acc)
    print(f"Epoch {epoch}: val_acc={acc:.3f}")

torch.save(model.state_dict(), "model.pt")

mlflow.pytorch.log_model(model, "model")
mlflow.end_run()



