import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocess import CatsDogsDataset
from model import SimpleCNN
import mlflow

train_ds = CatsDogsDataset("data/train")
val_ds = CatsDogsDataset("data/val")
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

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

torch.save(model, "model.pt")
mlflow.pytorch.log_model(model, "model")
mlflow.end_run()
