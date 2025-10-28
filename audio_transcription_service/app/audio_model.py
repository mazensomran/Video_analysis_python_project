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

        self.models_dir = Path("./models")
        self.models_dir.mkdir(exist_ok=True, parents=True)  # ✅ إنشاء المجلد إذا لم يكن موجوداً
        self.model_name = "base"
        self.device = device
        self.model = None
        self._load_model("base")

    def _load_model(self, model_name):
        """تحميل نموذج التعرف على الكلام"""
        try:
            logger.info(f"📥 تحميل نموذج التعرف على الكلام ({model_name}) على {self.device}...")
            available_models = MODEL_CONFIG["available_whisper_models"]
            if model_name not in available_models:
                logger.warning(f"⚠️ النموذج {model_name} غير متاح، استخدام 'base' بدلاً منه")
                model_name = "base"

            # ✅ تحميل النموذج مع معالجة أفضل للأخطاء
            self.model = whisper.load_model(
                model_name,
                download_root=str(self.models_dir / "whisper"),
                device=self.device
            )

            if self.model is not None:
                logger.info(f"✅ تم تحميل نموذج التعرف على الكلام على {self.device}")
            else:
                logger.error("❌ فشل في تحميل نموذج التعرف على الكلام")
                raise Exception("فشل في تحميل النموذج")

        except Exception as e:
            logger.error(f"❌ خطأ في تحميل نموذج التعرف على الكلام: {e}", exc_info=True)
            self.model = None
            raise  # ✅ إعادة رفع الاستثناء

    def transcribe_audio(self, video_path: str) -> dict:
        """تحويل الصوت إلى نص"""
        if self.model is None:
            logger.error("❌ النموذج غير محمل")
            return {"text": "", "language": "unknown", "confidence": 0.0}

        audio_path = None
        try:
            video_path_obj = Path(video_path)
            if not video_path_obj.exists():
                logger.error(f"❌ ملف الفيديو غير موجود: {video_path}")
                return {"text": "", "language": "unknown", "confidence": 0.0}

            logger.info(f"🎵 بدء معالجة الفيديو: {video_path}")

            # ✅ إنشاء ملف صوتي مؤقت
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_temp_file:
                audio_path = audio_temp_file.name

            logger.info("🔊 جاري استخراج الصوت من الفيديو...")
            if not self.extract_audio(video_path, audio_path):
                logger.error(f"❌ فشل في استخراج الصوت من الفيديو: {video_path}")
                return {"text": "", "language": "unknown", "confidence": 0.0}

            # ✅ التحقق من وجود الملف الصوتي وحجمه
            audio_file_size = Path(audio_path).stat().st_size
            if audio_file_size == 0:
                logger.error("❌ الملف الصوتي المستخرج فارغ")
                return {"text": "", "language": "unknown", "confidence": 0.0}

            logger.info(f"✅ تم استخراج الصوت بنجاح، الحجم: {audio_file_size} bytes")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("🎵 جاري تحويل الصوت إلى نص...")
            result = self.model.transcribe(audio_path, fp16=torch.cuda.is_available())
            logger.info(f"✅ تم التحويل بنجاح، النص: {len(result['text'])} حرف")

            # حساب الثقة ديناميكيًا
            confidence = 0.9
            if "segments" in result and result["segments"]:
                segment_confidences = [segment.get("avg_logprob", 0.0) for segment in result["segments"]]
                if segment_confidences:
                    confidence = sum(segment_confidences) / len(segment_confidences)

            transcription_result = {
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "confidence": confidence,
                "segments": result.get("segments", [])
            }

            logger.info(f"📊 نتيجة التحويل: لغة={transcription_result['language']}, ثقة={confidence:.2f}")
            return transcription_result

        except Exception as e:
            logger.error(f"❌ خطأ في معالجة الفيديو أو التحويل إلى نص: {e}", exc_info=True)
            return {"text": "", "language": "unknown", "confidence": 0.0}
        finally:
            # ✅ تنظيف الملف المؤقت
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                    logger.info("🧹 تم تنظيف الملف الصوتي المؤقت")
                except Exception as e:
                    logger.warning(f"⚠️ فشل في حذف الملف المؤقت: {e}")

    def extract_audio(self, video_path: str, output_audio_path: str) -> bool:
        """دالة لاستخراج الصوت من الفيديو مع التحقق من توفر FFmpeg"""
        try:
            # ✅ التحقق من توفر FFmpeg أولاً
            try:
                result = subprocess.run(['ffmpeg', '-version'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        timeout=10)
                if result.returncode != 0:
                    logger.error("❌ FFmpeg غير مثبت أو به مشكلة")
                    return False
                logger.info("✅ FFmpeg متوفر")
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                logger.error(f"❌ FFmpeg غير مثبت أو غير قابل للوصول: {e}")
                return False

            # ✅ استخدام أمر أكثر مرونة لاستخراج الصوت
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # لا فيديو
                '-acodec', 'pcm_s16le',  # ترميز صوتي مناسب
                '-ar', '16000',  # معدل العينة
                '-ac', '1',  # mono
                '-y',  # overwrite
                '-loglevel', 'error',
                output_audio_path
            ]

            logger.info(f"🔧 تشغيل أمر FFmpeg: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                logger.info(f"✅ تم استخراج الصوت بنجاح: {output_audio_path}")
                return True
            else:
                logger.error(f"❌ فشل في استخراج الصوت: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ انتهت المهلة في استخراج الصوت")
            return False
        except Exception as e:
            logger.error(f"❌ خطأ في استخراج الصوت: {e}")
            return False

    def cleanup(self):
        """تنظيف الموارد"""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("🧹 تم تنظيف موارد SpeechRecognizer")
