from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os


from activity_model import ActivityRecognizer, MarianTranslator

app = FastAPI(title="خدمة تحليل النشاط والبيئة")

# تحميل النماذج
activity_recognizer = ActivityRecognizer()
translator = MarianTranslator()


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

        activity_analysis_ar = translator.translate(activity_analysis_en)

        os.unlink(temp_path)

        return {
            "activity_analysis_en": activity_analysis_en,
            "activity_analysis_ar": activity_analysis_ar
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في المعالجة: {str(e)}")

