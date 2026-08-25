import os
import logging
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator

from src.model import SimpleCNN

app = FastAPI()
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))


model = SimpleCNN()
checkpoint_path = Path(__file__).resolve().parent.parent / "model.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
if isinstance(checkpoint, torch.nn.Module):
    model.load_state_dict(checkpoint.state_dict())
else:
    model.load_state_dict(checkpoint)
model.eval()

transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

logging.basicConfig(level=logging.INFO)

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    started_at = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - started_at) * 1000
    logging.info(
        "request method=%s path=%s status=%s latency_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile):
    image = Image.open(file.file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]
    confidence, class_index = probabilities.max(0)
    prediction = "cat" if class_index.item() == 0 else "dog"
    if confidence.item() < CONFIDENCE_THRESHOLD:
        prediction = "unknown"
    logging.info("prediction class=%s confidence=%.3f", prediction, confidence.item())
    return {"prediction": prediction, "confidence": round(confidence.item(), 4)}

Instrumentator().instrument(app).expose(app)

@app.get("/", response_class=HTMLResponse)
def main():
    content = """
    <html>
        <body>
            <h2>Cats vs Dogs Classifier</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit" value="Upload and Predict">
            </form>
        </body>
    </html>
    """
    return content
