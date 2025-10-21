import cv2
import numpy as np
import torch
import easyocr
from pathlib import Path
from typing import List, Dict, Any
import logging
from shared.config import PROCESSING_CONFIG

logger = logging.getLogger(__name__)


class TextDetector:
    def __init__(self, text_threshold=0.3):
        self.reader = easyocr.Reader(['en', 'ar'], gpu=(self.device == "gpu"))
        self.detection_threshold = text_threshold
        self.min_text_confidence = 0.3
        self.languages = ['ar', 'en']
        self.reader = None
        self.frame_counter = 0
        self.enabled = PROCESSING_CONFIG["text_detection_enabled"]
        if self.enabled:
            self._setup_easyocr()

    def _setup_easyocr(self):
        """إعداد EasyOCR"""
        try:
            from shared.config import EASYOCR_CONFIG
            logger.info("📥 جاري تحميل EasyOCR...")

            gpu = EASYOCR_CONFIG["gpu_enabled"] and torch.cuda.is_available()
            model_dir = Path(EASYOCR_CONFIG["model_storage_directory"])
            model_dir.mkdir(parents=True, exist_ok=True)

            # ✅ التصحيح: استخدام المعلمات الصحيحة
            self.reader = easyocr.Reader(
                lang_list=self.languages,
                gpu=gpu,
                model_storage_directory=str(model_dir),
                download_enabled=EASYOCR_CONFIG["download_enabled"],
                detector=EASYOCR_CONFIG["detector"],  # ✅ تصحيح اسم المعلمة
                recognizer=EASYOCR_CONFIG["recognizer"]  # ✅ تصحيح اسم المعلمة
            )

            logger.info(f"✅ تم تحميل EasyOCR بنجاح على {'GPU' if gpu else 'CPU'}")

        except Exception as e:
            logger.error(f"❌ فشل في تحميل EasyOCR: {e}")
            self.reader = None
            self.enabled = False

    def detect_text(self, frame: np.ndarray, threshold: float =None) -> List[Dict[str, Any]]:
        """كشف النص في إطار"""
        if self.reader is None:
            return []

        # أخذ عينة من الإطارات فقط
        self.frame_counter += 1

        try:
            # تحسين الصورة لتحسين دقة التعرف على النص
            enhanced_frame = self._enhance_image_for_text(frame)
            results = self.reader.readtext(enhanced_frame, paragraph=False)
            current_threshold = threshold if threshold is not None else self.min_text_confidence

            text_data = []
            for (bbox, text, confidence) in results:
                if confidence >= current_threshold:
                    points = np.array(bbox).astype(int)
                    x1, y1 = np.min(points[:, 0]), np.min(points[:, 1])
                    x2, y2 = np.max(points[:, 0]), np.max(points[:, 1])
                    width, height = x2 - x1, y2 - y1

                    # تجاهل النصوص الصغيرة جداً
                    if width < 10 or height < 10:
                        continue

                    language = self._detect_language(text)
                    text_data.append({
                        "bbox": [int(x1), int(y1), int(width), int(height)],
                        "text": text.strip(),
                        "confidence": float(confidence),
                        "language": language
                    })

            return text_data

        except Exception as e:
            logger.error(f"❌ خطأ في كشف النص: {e}")
            return []

    def _enhance_image_for_text(self, frame: np.ndarray) -> np.ndarray:
        """تحسين الصورة لتحسين التعرف على النص"""
        try:
            # تحويل إلى تدرجات الرمادي
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # تطبيق تصحيح التباين
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # تحويل مرة أخرى إلى BGR
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        except Exception:
            return frame

    def _detect_language(self, text: str) -> str:
        """كشف اللغة تلقائياً بناء على النص"""
        arabic_chars = "ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوي"

        if any(char in arabic_chars for char in text):
            return "ar"
        elif any(char.isalpha() for char in text):
            return "en"
        return "unknown"

    def cleanup(self):
        """تنظيف ذاكرة EasyOCR"""
        if self.reader:
            try:
                self.reader = None
                logger.info("🧹 تم تنظيف ذاكرة EasyOCR")
            except Exception as e:
                logger.error(f"⚠️ خطأ في تنظيف EasyOCR: {e}")