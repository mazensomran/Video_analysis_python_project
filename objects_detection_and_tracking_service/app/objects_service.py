from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from typing import Dict, Any, List
import tempfile
import os
import base64
import json
from fastapi.encoders import jsonable_encoder

from objects_model import ObjectTracker

app = FastAPI(title="خدمة كشف وتتبع الكائنات")

# تحميل النموذج
object_tracker = ObjectTracker()

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), threshold: float = 0.5):

    try:
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

        # تنظيف الملف المؤقت
        os.unlink(temp_path)

        return {"objects": objects_data, "count": len(objects_data)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في المعالجة: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "object", "model_loaded": model is not None}
