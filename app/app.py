import os
import pathlib
import mimetypes
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import easyocr

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.model_construction.PlateLocNet_Classifier import PlateLocNet

mimetypes.add_type('text/css', '.css')
mimetypes.add_type('application/javascript', '.js')

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CURRENT_FILE_PATH = pathlib.Path(__file__).resolve()
APP_DIR = CURRENT_FILE_PATH.parent
PROJECT_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR

MODEL_PATH = PROJECT_ROOT / "src" / "models" / "PlateLocNet.pth"

if not STATIC_DIR.exists():
    print(f"[ERROR] Application directory does not exist: {STATIC_DIR}")
else:
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

class LicensePlatePipeline:
    def __init__(self, model_path, device):
        self.device = device

        print("[1/2] Loading PlateLocNet localization model...")
        self.model_loc = PlateLocNet()
        if os.path.exists(model_path):
            self.model_loc.load_state_dict(torch.load(str(model_path), map_location=self.device))
        else:
            print(f"[WARNING] Missing model weights file at path: {model_path}")
        self.model_loc.to(self.device)
        self.model_loc.eval()

        print("[2/2] Initializing EasyOCR reader...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=(self.device == "cuda"))
        print("[STATUS] Pipeline models ready for operation!")

        self.detection_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def _convert_yolo_pred_to_bbox_px(self, prediction, orig_w, orig_h):
        pred_confidence = prediction[..., 0]
        best_cell = torch.argmax(pred_confidence)
        cell_y = int(best_cell // 14)
        cell_x = int(best_cell % 14)

        highest_conf = pred_confidence[cell_y, cell_x].item()

        if highest_conf < 0.4:
            return None, highest_conf

        pred_box = prediction[cell_y, cell_x, 1:5]

        x_global = (cell_x + pred_box[0].item()) / 14.0
        y_global = (cell_y + pred_box[1].item()) / 14.0
        w_global = pred_box[2].item()
        h_global = pred_box[3].item()

        xmin = int((x_global - w_global / 2.0) * orig_w)
        ymin = int((y_global - h_global / 2.0) * orig_h)
        xmax = int((x_global + w_global / 2.0) * orig_w)
        ymax = int((y_global + h_global / 2.0) * orig_h)

        bbox = [max(0, xmin), max(0, ymin), min(orig_w, xmax), min(orig_h, ymax)]

        relative_bbox = {
            "left": max(0.0, x_global - w_global / 2.0),
            "top": max(0.0, y_global - h_global / 2.0),
            "width": w_global,
            "height": h_global
        }

        return bbox, relative_bbox, highest_conf

    @torch.no_grad()
    def process_image(self, orig_img):
        orig_h, orig_w, _ = orig_img.shape
        img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        augmented = self.detection_transform(image=img_rgb)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)

        prediction = self.model_loc(input_tensor).squeeze(0)

        conversion_result = self._convert_yolo_pred_to_bbox_px(prediction, orig_w, orig_h)
        if conversion_result[0] is None:
            conf_loc = conversion_result[1]
            return {
                "plate": "Not detected",
                "confidence": 0.0,
                "detection_confidence": round(conf_loc * 100, 1),
                "bbox": None
            }

        bbox_px, relative_bbox, conf_loc = conversion_result
        xmin, ymin, xmax, ymax = bbox_px

        margin_x = int((xmax - xmin) * 0.03)
        margin_y = int((ymax - ymin) * 0.05)

        crop_xmin = max(0, xmin - margin_x)
        crop_ymin = max(0, ymin - margin_y)
        crop_xmax = min(orig_w, xmax + margin_x)
        crop_ymax = min(orig_h, ymax + margin_y)

        cropped_plate = orig_img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        ocr_results = self.ocr_reader.readtext(cropped_plate)
        cleaned_text = "NO READ"
        ocr_confidence = 0.0

        if len(ocr_results) > 0:
            best_ocr = max(ocr_results, key=lambda x: x[2])
            cleaned_text = "".join(best_ocr[1].split()).upper()
            ocr_confidence = round(float(best_ocr[2]) * 100, 1)

        return {
            "plate": cleaned_text,
            "confidence": ocr_confidence,
            "detection_confidence": round(conf_loc * 100, 1),
            "bbox": relative_bbox
        }

pipeline = LicensePlatePipeline(MODEL_PATH, DEVICE)

@app.get("/")
async def serve_index():
    return FileResponse(APP_DIR / "index.html")

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Unsupported or corrupted image file extension.")

        results = pipeline.process_image(img_bgr)

        return {
            "plate": results["plate"],
            "confidence": results["confidence"],
            "detection_confidence": results["detection_confidence"],
            "bbox": results["bbox"]
        }

    except Exception as e:
        print(f"[SERVER ERROR]: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)