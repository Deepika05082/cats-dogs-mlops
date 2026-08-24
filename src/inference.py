from fastapi import FastAPI, UploadFile
from fastapi import Request
import torch
from PIL import Image
import torchvision.transforms as T
import logging
import time
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import HTMLResponse
from src.model import SimpleCNN   # keep only one import
from pathlib import Path

app = FastAPI()

# Load both the current state-dict format and older full-model checkpoints.
model = SimpleCNN()
checkpoint_path = Path(__file__).resolve().parent.parent / "model.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
if isinstance(checkpoint, torch.nn.Module):
    model.load_state_dict(checkpoint.state_dict())
else:
    model.load_state_dict(checkpoint)
model.eval()

transform = T.Compose([T.Resize((224,224)), T.ToTensor()])

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
    img = Image.open(file.file).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        y = model(x).argmax().item()
    prediction = "cat" if y == 0 else "dog"
    logging.info("prediction class=%s", prediction)
    return {"prediction": prediction}

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
