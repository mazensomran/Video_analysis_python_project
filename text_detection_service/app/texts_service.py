from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import cv2
import tempfile
import os
from texts_model import TextDetector
import logging
from fastapi.middleware.cors import CORSMiddleware
# إعداد التسجيل
logging.basicConfig(level=logging.INFO)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(title="خدمة استخراج النصوص")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل النموذج مع معالجة الأخطاء
try:
    text_detector = TextDetector()
    logger.info("✅ تم تحميل نموذج اكتشاف النصوص بنجاح")
except Exception as e:
    logger.error(f"❌ فشل في تحميل نموذج اكتشاف النصوص: {e}")
    text_detector = None

@app.post("/detect")
async def detect_text(file: UploadFile = File(...),
                      threshold: float = Form(0.5) ):
    if text_detector is None:
        raise HTTPException(status_code=500, detail="نموذج اكتشاف النصوص غير متاح")

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
        results = text_detector.detect_text(image, threshold=threshold)

        # تنظيف الملف المؤقت
        os.unlink(temp_path)

        return {"texts": results, "count": len(results)}

    except Exception as e:
        logger.error(f"❌ خطأ في معالجة النص: {e}")
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
    status = "healthy" if text_detector and text_detector.enabled else "unhealthy"
    return {
        "status": status,
        "service": "text_detection",
        "model_loaded": text_detector is not None and text_detector.enabled
    }

@app.get("/")
async def root():
    return {"message": "خدمة اكتشاف النصوص تعمل", "status": "active"}
