import logging
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import cv2
import tempfile
import os
from fastapi.middleware.cors import CORSMiddleware

from objects_model import ObjectTracker

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(title="خدمة كشف وتتبع الكائنات")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# تحميل النموذج
object_tracker = ObjectTracker()

def convert_serializable_types(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {key: convert_serializable_types(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_serializable_types(item) for item in obj]
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist() if obj.numel() > 1 else float(obj.item())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), threshold: float = Form(0.5)):
    temp_path = None
    try:
        tracks = []
        # حفظ الملف مؤقتاً
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # قراءة الصورة
        image = cv2.imread(temp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="تعذر قراءة الصورة")

        # تحويل الصورة
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        threshold = threshold
        objects_data = object_tracker.track_objects(rgb_image, threshold)
        objects_data = convert_serializable_types(objects_data)

        # فصل الأشخاص عن الكائنات الأخرى
        person_tracks = []
        other_objects = []

        for det in objects_data:
            if det["class_name"] == "person" and det.get("track_id") is not None:
                person_tracks.append(det)
            else:
                other_objects.append(det)

        # تنظيف الملف المؤقت
        os.unlink(temp_path)

        return {
            "objects": other_objects,
            "tracks": person_tracks,
            "count": len(objects_data)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في المعالجة: {str(e)}")
    finally:
        # تنظيف الملف المؤقت في جميع الحالات
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"⚠️ فشل في تنظيف الملف المؤقت: {e}")


@app.get("/health")
async def health_check():
    """فحص صحة الخدمة"""
    status = "healthy" if object_tracker and object_tracker.model else "unhealthy"
    return {
        "status": status,
        "service": "activity-analysis",
        "model_loaded": object_tracker is not None and object_tracker.model is not None
    }


@app.get("/")
async def root():
    return {"message": "خدمة كشف وتتبع الكائنات تعمل"}
