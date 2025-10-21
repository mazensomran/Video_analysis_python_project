import subprocess
import torch
import whisper
from pathlib import Path
import logging
from shared.config import MODEL_CONFIG
import tempfile
import os

logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SpeechRecognizer:
    def __init__(self):
        self.models_dir =  Path("../")
        self.model_name = "base"
        self.device = device
        self.model = None
        self._load_model("base")

    def _load_model(self, model_name):
        """تحميل نموذج التعرف على الكلام"""
        try:
            logger.info(f"📥 تحميل نموذج التعرف على الكلام ({self.model_name}) على {self.device}...")
            available_models = MODEL_CONFIG["available_whisper_models"]
            if model_name not in available_models:
                print(f"⚠️ النموذج {model_name} غير متاح، استخدام 'base' بدلاً منه")
                model_name = "base"

            self.model = whisper.load_model(
                model_name,
                download_root=str(self.models_dir / "whisper"),
                device=self.device
            )

            if self.model is not None:
                logger.info(f"✅ تم تحميل نموذج التعرف على الكلام على {self.device}")
            else:
                logger.error("❌ فشل في تحميل نموذج التعرف على الكلام")

        except Exception as e:
            logger.error(f"❌ خطأ في تحميل نموذج التعرف على الكلام: {e}")
            self.model = None

    def transcribe_audio(self, video_path: str) -> dict:
        """تحويل الصوت إلى نص"""
        if self.model is None:
            return {"text": "", "language": "unknown", "confidence": 0.0}

        try:
            if not Path(video_path).exists():
                logger.error(f"❌ ملف الفيديو غير موجود: {video_path}")
                return {"text": "", "language": "unknown", "confidence": 0.0}

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_temp_file:
                audio_path = audio_temp_file.name

            if not self.extract_audio(video_path, audio_path):
                logger.error(f"❌ فشل في استخراج الصوت من الفيديو: {video_path}")
                return {"text": "", "language": "unknown", "confidence": 0.0}

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("🎵 جاري تحويل الصوت إلى نص...")
            result = self.model.transcribe(audio_path, fp16=torch.cuda.is_available())

            # حساب الثقة ديناميكيًا
            confidence = 0.9  # قيمة افتراضية
            if "segments" in result and result["segments"]:
                segment_confidences = [segment.get("avg_logprob", 0.0) for segment in
                                       result["segments"]]  # مثال: استخدام avg_logprob
                if segment_confidences:
                    confidence = sum(segment_confidences) / len(segment_confidences)  # متوسط الثقة

            transcription_result = {
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "confidence": confidence,  # الآن ديناميكي
                "segments": result.get("segments", [])
            }
            return transcription_result
        except Exception as e:
            logger.error(f"❌ خطأ في معالجة الفيديو أو التحويل إلى نص: {e}")
            return {"text": "", "language": "unknown", "confidence": 0.0}
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def extract_audio(self, video_path: str, output_audio_path: str) -> bool:
        """دالة لاستخراج الصوت من الفيديو مع التحقق من توفر FFmpeg"""
        try:
            # التحقق من توفر FFmpeg أولاً
            try:
                subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except FileNotFoundError:
                logger.error("❌ FFmpeg غير مثبت. يرجى التأكد من تثبيته.")
                return False

            command = [
                'ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a',
                output_audio_path, '-y', '-loglevel', 'error'
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ تم استخراج الصوت بنجاح: {output_audio_path}")
                return True
            else:
                logger.error(f"❌ فشل في استخراج الصوت: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ خطأ في استخراج الصوت: {e}")
            return False

    def cleanup(self):
        """تنظيف الموارد"""
        self.model = None
        logger.info("🧹 تم تنظيف موارد SpeechRecognizer")

