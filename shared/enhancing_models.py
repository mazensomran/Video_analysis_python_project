import os
import tempfile
import cv2
import numpy as np
from pathlib import Path
import logging
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


class FrameEnhancer:
    def __init__(self, brightness=1.0, contrast=1.0, saturation=1.0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def enhance_frame(self, frame):
        if frame is None or frame.size == 0:
            logger.warning("⚠️ تم تمرير إطار فارغ إلى FrameEnhancer.enhance_frame.")
            return None


        # تحويل BGR إلى RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # تحويل numpy array إلى PIL Image
        img_pil = Image.fromarray(img_rgb)
        # تطبيق تحسينات السطوع والتباين والتشبع
        if self.brightness != 1.0:
            img_pil = ImageEnhance.Brightness(img_pil).enhance(self.brightness)
        if self.contrast != 1.0:
            img_pil = ImageEnhance.Contrast(img_pil).enhance(self.contrast)
        if self.saturation != 1.0:
            img_pil = ImageEnhance.Color(img_pil).enhance(self.saturation)
        return img_pil


class VideoEnhancer:
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def enhance_video(self, input_path: str, strength: int = 2) -> str:

        try:
            if strength < 1 or strength > 5:
                strength = 2

            print(f"🎨 بدء تحسين الفيديو بمستوى قوة: {strength}")

            # فتح الفيديو
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print("❌ تعذر فتح الفيديو للتحسين")
                return input_path

            # الحصول على مواصفات الفيديو
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"📊 مواصفات الفيديو: {width}x{height}, {fps} FPS, {total_frames} إطار")

            # إذا كان الفيديو عالي الجودة بالفعل، تخطي التحسين
            if width >= 1280 and height >= 720 and strength < 3:
                print("✅ الفيديو عالي الجودة بالفعل، تخطي التحسين")
                cap.release()
                return input_path

            # إنشاء ملف مؤقت للفيديو المحسن
            temp_dir = tempfile.gettempdir()
            enhanced_path = os.path.join(temp_dir, f"enhanced_{os.path.basename(input_path)}")

            # إعداد كوديك الفيديو
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(enhanced_path, fourcc, fps, (width, height))

            frames_processed = 0
            frame_skip = self._get_frame_skip(strength, total_frames)

            print(f"⏩ تخطي الإطارات: كل {frame_skip} إطارات")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frames_processed += 1

                # تخطي بعض الإطارات للسرعة (للstrength المنخفض)
                if frame_skip > 1 and frames_processed % frame_skip != 0:
                    out.write(frame)
                    continue

                # تطبيق التحسينات
                enhanced_frame = self._enhance_frame(frame, strength)
                out.write(enhanced_frame)

                if frames_processed % 100 == 0:
                    print(f"🔄 معالجة {frames_processed}/{total_frames} إطار")

            cap.release()
            out.release()

            print(f"✅ تم تحسين الفيديو بنجاح: {enhanced_path}")
            return enhanced_path

        except Exception as e:
            print(f"❌ خطأ في تحسين الفيديو: {str(e)}")
            # في حالة الخطأ، ارجع الفيديو الأصلي
            return input_path

    def _get_frame_skip(self, strength: int, total_frames: int) -> int:
        """تحديد عدد الإطارات التي يتم تخطيها بناءً على القوة والطول"""
        if total_frames > 1000:
            # للفيديوهات الطويلة، تخطي المزيد من الإطارات للسرعة
            skip_map = {1: 3, 2: 2, 3: 1, 4: 1, 5: 1}
        else:
            # للفيديوهات القصيرة، معالجة معظم الإطارات
            skip_map = {1: 2, 2: 1, 3: 1, 4: 1, 5: 1}

        return skip_map.get(strength, 1)

    def _enhance_frame(self, frame: np.ndarray, strength: int) -> np.ndarray:
        """
        تحسين إطار فردي باستخدام طرق سريعة
        """
        enhanced = frame.copy()

        try:
            # 1. تحسين حدة الصورة (سريع)
            if strength >= 2:
                kernel = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)

            # 2. تحسين التباين باستخدام CLAHE (سريع وفعال)
            if strength >= 3:
                # تحويل إلى مساحة لون LAB
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)

                # تطبيق CLAHE على قناة L (اللمعان)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_enhanced = clahe.apply(l)

                # دمج القنوات مرة أخرى
                lab_enhanced = cv2.merge([l_enhanced, a, b])
                enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

            # 3. تقليل الضوضاء (للstrength العالي)
            if strength >= 4:
                enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 3, 9)

            # 4. تحسين السطوع والتباين البسيط (للstrength المنخفض)
            if strength == 1:
                alpha = 1.1  # عامل التباين
                beta = 5  # عامل السطوع
                enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)

            return enhanced

        except Exception as e:
            print(f"⚠️ خطأ في تحسين الإطار: {str(e)}")
            return frame

    def cleanup_temp_files(self):
        """تنظيف الملفات المؤقتة"""
        temp_dir = tempfile.gettempdir()
        for file in Path(temp_dir).glob("enhanced_*"):
            try:
                file.unlink()
                print(f"🧹 تم حذف الملف المؤقت: {file}")
            except Exception as e:
                print(f"⚠️ تعذر حذف الملف المؤقت: {file} - {e}")


# إنشاء نسخة عامة
video_enhancer = VideoEnhancer()









