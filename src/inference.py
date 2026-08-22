from fastapi import FastAPI, UploadFile
import torch
from PIL import Image
import torchvision.transforms as T
import logging
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
model = torch.load("model.pt")
model.eval()
transform = T.Compose([T.Resize((224,224)), T.ToTensor()])

logging.basicConfig(level=logging.INFO)

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile):
    logging.info(f"Received request: {file.filename}")
    img = Image.open(file.file).convert("RGB")
    x = transform(img).unsqueeze(0)
    y = model(x).argmax().item()
    logging.info(f"Prediction: {y}")
    return {"prediction": "cat" if y==0 else "dog"}

Instrumentator().instrument(app).expose(app)
