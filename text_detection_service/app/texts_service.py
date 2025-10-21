from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import cv2
import tempfile
import os
from texts_model import TextDetector

app = FastAPI(title="خدمة استخراج النصوص")

# تحميل النموذج
text_detector = TextDetector()
@app.post("/detect")
async def detect_text(
    file: UploadFile = File(...),
    threshold: float = Form(0.3)  # إضافة threshold
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

        # كشف النصوص
        results = text_detector.detect_text(image, threshold = threshold)

        # تنظيف الملف المؤقت
        os.unlink(temp_path)

        return {"texts": results, "count": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في المعالجة: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "text"}
