import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
from src.model import SimpleCNN
from src.preprocess import CatsDogsDataset
import mlflow
from torch.utils.data import DataLoader, random_split

DATA_PATH = "data/PetImages"
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001

full_ds = CatsDogsDataset(DATA_PATH)
train_size = int(0.8 * len(full_ds))
val_size = len(full_ds) - train_size

train_ds, val_ds = random_split(full_ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)


model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
val_losses = []
val_accuracies = []
confusion_matrix = torch.zeros(2, 2, dtype=torch.int64)

with mlflow.start_run(run_name="cats-dogs-cnn"):
    mlflow.log_params({
        "data_path": DATA_PATH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_samples": train_size,
        "validation_samples": val_size,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
    })

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)

        model.eval()
        confusion_matrix.zero_()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                predictions = outputs.argmax(1)
                val_loss += loss.item() * labels.size(0)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                for actual, predicted in zip(labels, predictions):
                    confusion_matrix[actual, predicted] += 1

        train_loss /= train_size
        val_loss /= val_size
        accuracy = correct / total
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": accuracy,
        }, step=epoch)
        print(f"Epoch {epoch}: train_loss={train_loss:.3f}, val_loss={val_loss:.3f}, val_acc={accuracy:.3f}")

    torch.save(model.state_dict(), "model.pt")

    with tempfile.TemporaryDirectory() as artifact_dir:
        artifact_path = Path(artifact_dir)

        plt.figure()
        plt.plot(range(1, EPOCHS + 1), train_losses, label="train_loss")
        plt.plot(range(1, EPOCHS + 1), val_losses, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        loss_curve_path = artifact_path / "loss_curves.png"
        plt.savefig(loss_curve_path)
        plt.close()

        plt.figure()
        plt.imshow(confusion_matrix.numpy(), cmap="Blues")
        plt.xticks([0, 1], ["cat", "dog"])
        plt.yticks([0, 1], ["cat", "dog"])
        plt.xlabel("Predicted label")
        plt.ylabel("Actual label")
        for row in range(2):
            for column in range(2):
                plt.text(column, row, confusion_matrix[row, column].item(), ha="center", va="center")
        plt.colorbar()
        plt.tight_layout()
        confusion_matrix_path = artifact_path / "confusion_matrix.png"
        plt.savefig(confusion_matrix_path)
        plt.close()

        mlflow.log_artifact(str(loss_curve_path), artifact_path="plots")
        mlflow.log_artifact(str(confusion_matrix_path), artifact_path="plots")
    mlflow.pytorch.log_model(model, "model")

