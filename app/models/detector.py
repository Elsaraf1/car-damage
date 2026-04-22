from ultralytics import YOLO
from PIL import Image
import numpy as np

def load_detector(weights_path: str):
    return YOLO(weights_path)

def run_detector(model, image: Image.Image) -> Image.Image:
    results = model.predict(
        source=np.array(image.convert("RGB")),
        imgsz=640,
        conf=0.25,
        verbose=False,
    )
    # plot() returns a BGR numpy array — convert to RGB PIL
    annotated = results[0].plot()
    return Image.fromarray(annotated[..., ::-1])
