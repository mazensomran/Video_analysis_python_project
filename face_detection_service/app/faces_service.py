from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import tempfile
import os
import logging


from faces_model import FaceDetector

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(title="خدمة كشف الوجوه في الصور والفيديوهات")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    status = "healthy" if detector and detector.scrfd_detector else "unhealthy"
    return {
        "status": status,
        "service": "activity-analysis",
        "model_loaded": detector is not None and detector.scrfd_detector is not None
    }


@app.get("/")
async def root():
    return {"message": "خدمة كشف الوجوه تعمل"}
