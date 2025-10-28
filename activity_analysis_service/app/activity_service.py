import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os

from activity_model import ActivityRecognizer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(title="خدمة تحليل النشاط والبيئة")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# تحميل النماذج
activity_recognizer = ActivityRecognizer()


@app.post("/analyze")
async def analyze_activity(
        file: UploadFile = File(...),
        prompt: str = Form("Describe the activities in this video"),
        # إضافة المعاملات المتقدمة
        max_new_tokens: int = Form(600),
        temperature: float = Form(0.3),
        top_p: float = Form(0.9),
        top_k: int = Form(50),
        do_sample: bool = Form(True),
        enable_enhancement: bool = Form(False),
        enhancement_strength: int = Form(2),
        fps: float = Form(1.0),
        pixels_size: int = Form(512)
):
    try:
        # حفظ الملف مؤقتاً
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # تمرير جميع المعاملات للrecognize_activity
        activity_analysis_en = activity_recognizer.recognize_activity(
            prompt=prompt,
            video_path=temp_path,
            fsp=fps,
            pixels_size=pixels_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            enable_enhancement=enable_enhancement,
            enhancement_strength=enhancement_strength
        )


        os.unlink(temp_path)

        return {
            "activity_analysis_en": activity_analysis_en,
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
    status = "healthy" if activity_recognizer and activity_recognizer.qwen2_vl_model else "unhealthy"
    return {
        "status": status,
        "service": "activity-analysis",
        "model_loaded": activity_recognizer is not None and activity_recognizer.qwen2_vl_model is not None
    }


@app.get("/")
async def root():
    return {"message": "خدمة تحليل النشاط والبيئة تعمل"}

