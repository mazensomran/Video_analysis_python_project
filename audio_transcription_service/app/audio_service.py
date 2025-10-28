from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os
import logging
from fastapi.middleware.cors import CORSMiddleware
from audio_model import SpeechRecognizer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

setup_logging()

logger = logging.getLogger(__name__)

# ✅ إعداد التسجيل
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="خدمة تحويل الصوت إلى نص")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ تحميل النموذج مع معالجة الأخطاء
try:
    speech_recognizer = SpeechRecognizer()
    logger.info("✅ تم تحميل نموذج التعرف على الكلام بنجاح")
except Exception as e:
    logger.error(f"❌ فشل في تحميل النموذج: {e}")
    speech_recognizer = None


@app.post("/transcribe")
async def transcribe_video(file: UploadFile = File(..., max_size=200 * 1024 * 1024)):
    """تحويل الفيديو إلى نص"""

    # ✅ التحقق من حالة النموذج
    if speech_recognizer is None or speech_recognizer.model is None:
        logger.error("❌ خدمة التعرف على الكلام غير جاهزة")
        raise HTTPException(status_code=503, detail="خدمة التعرف على الكلام غير جاهزة")

    # ✅ التحقق من نوع الملف
    allowed_content_types = ['video/', 'audio/', 'application/octet-stream']
    if not any(file.content_type.startswith(ct) for ct in allowed_content_types):
        logger.error(f"❌ نوع ملف غير مدعوم: {file.content_type}")
        raise HTTPException(status_code=400, detail="الملف يجب أن يكون فيديو أو صوتي")

    temp_video_path = None
    try:
        # ✅ حفظ الملف المحمل مؤقتًا
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            content = await file.read()
            temp_video_file.write(content)
            temp_video_path = temp_video_file.name

        logger.info(f"📁 تم حفظ الملف المؤقت: {temp_video_path}، الحجم: {len(content)} bytes")

        # ✅ استخدام الدالة لمعالجة الفيديو
        result = speech_recognizer.transcribe_audio(temp_video_path)

        # ✅ التحقق من نتيجة التحويل
        if result.get("text", "").strip() and result.get("confidence", 0) > 0:
            logger.info(f"✅ تم التحويل بنجاح، طول النص: {len(result['text'])}")
            return JSONResponse(result)
        else:
            logger.warning("⚠️ التحويل لم ينتج نصاً أو الثقة منخفضة")
            return JSONResponse(result)

    except Exception as e:
        logger.error(f"❌ خطأ في معالجة الفيديو: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطأ في معالجة الفيديو: {str(e)}")
    finally:
        # ✅ تنظيف الملف المؤقت
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
                logger.info("🧹 تم تنظيف الملف المؤقت")
            except Exception as e:
                logger.warning(f"⚠️ فشل في تنظيف الملف المؤقت: {e}")


@app.get("/health")
async def health_check():
    """فحص صحة الخدمة"""
    status = "healthy" if speech_recognizer and speech_recognizer.model else "unhealthy"
    return {
        "status": status,
        "service": "audio-transcription",
        "model_loaded": speech_recognizer is not None and speech_recognizer.model is not None
    }


@app.get("/")
async def root():
    return {"message": "خدمة تحويل الصوت إلى نص تعمل"}

