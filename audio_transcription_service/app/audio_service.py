from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os

from audio_model import SpeechRecognizer


app = FastAPI(title="خدمة تحويل الصوت إلى نص")

# تحميل النموذج
speech_recognizer = SpeechRecognizer()

@app.post("/transcribe")
async def transcribe_video(file: UploadFile = File(...,max_size=200 * 1024 * 1024)) -> JSONResponse:

    try:
        # التحقق من نوع الملف (قبول الفيديو مثل mp4)
        if not file.content_type.startswith(('video/', 'audio/')):  # لدعم الفيديو أيضًا
            raise HTTPException(status_code=400, detail="الملف يجب أن يكون فيديو أو صوتي")
        # حفظ الملف المحمل مؤقتًا
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:  # يمكن تغيير الصيغة
            content = await file.read()
            temp_video_file.write(content)
            video_path = temp_video_file.name  # مسار الفيديو المؤقت
        try:
            # استخدام الدالة الجديدة لمعالجة الفيديو
            result = speech_recognizer.transcribe_audio(video_path)  # الدالة الجديدة
            return result
            '''return JSONResponse({
                "status": "success",
                "transcription": result,
                "message": "تم تحويل الفيديو إلى نص بنجاح"
            })'''
        finally:
            # تنظيف الملف الفيديو المؤقت
            if os.path.exists(video_path):
                os.unlink(video_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في معالجة الفيديو أو التحويل إلى نص: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "audio-transcription"}


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8001
    print(f"الواجهة التفاعلية (Swagger UI): http://{host}:{port}/docs")
    print(f"الواجهة البديلة (ReDoc): http://{host}:{port}/redoc")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
