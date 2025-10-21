from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import cv2
import tempfile
import os
import logging


from faces_model import FaceDetector

logger = logging.getLogger(__name__)

app = FastAPI(title="خدمة كشف الوجوه في الصور والفيديوهات")

detector = FaceDetector()

@app.post("/detect")
async def detect_faces(
    file: UploadFile = File(...),
    threshold: float = Form(0.5)  # إضافة threshold
):

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

        # كشف الوجوه
        threshold = threshold
        faces = detector.detect_faces(image, threshold=threshold)

        # تنظيف الملف المؤقت
        os.unlink(temp_path)

        return {"faces": faces, "count": len(faces)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في المعالجة: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "face", "model_loaded": detector is not None}
