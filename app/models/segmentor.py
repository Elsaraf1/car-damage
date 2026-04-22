from ultralytics import YOLO
from PIL import Image
import numpy as np

def load_segmentor(weights_path: str):
    return YOLO(weights_path)

def run_segmentor(model, image: Image.Image) -> Image.Image:
    results = model.predict(
        source=np.array(image.convert("RGB")),
        imgsz=640,
        conf=0.25,
        verbose=False,
    )
    annotated = results[0].plot()
    return Image.fromarray(annotated[..., ::-1])
