import os
import torch
import logging
from shared.enhancing_models import VideoEnhancer
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
import traceback

# تعطيل التحذيرات غير الضرورية
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

logger = logging.getLogger(__name__)
video_enhancer = VideoEnhancer()


def move_to_device(obj, device):
    """نقل كل tensors داخل dict/list/tensor إلى نفس الجهاز"""
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    else:
        return obj


class ActivityRecognizer:
    def __init__(self):
        try:
            # التحقق من توفر GPU أولاً
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("🎯 تم اكتشاف GPU، جاري التحميل على CUDA...")
            else:
                self.device = torch.device("cpu")
                logger.info("⚙️  لم يتم اكتشاف GPU، جاري التحميل على CPU...")

            # تحميل المعالج أولاً
            logger.info("📥 جاري تحميل معالج Qwen2-VL...")
            self.qwen2_vl_proc = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct",
                trust_remote_code=True
            )

            # ثم تحميل النموذج
            logger.info("📥 جاري تحميل نموذج Qwen2-VL...")
            self.qwen2_vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )

            # وضع النموذج في وضع التقييم
            self.qwen2_vl_model.eval()

            logger.info(f"✅ تم تهيئة ActivityRecognizer بنجاح على {self.device}")

        except Exception as e:
            logger.error(f"❌ فشل تهيئة ActivityRecognizer: {e}")
            logger.error(traceback.format_exc())
            self.qwen2_vl_model = None
            self.qwen2_vl_proc = None
            self.device = None

    def recognize_activity(self, prompt: str, video_path: str, fsp: float, pixels_size: int,
                           max_new_tokens: int = 600, temperature: float = 0.3,
                           top_p: float = 0.9, top_k: int = 50, do_sample: bool = True,
                           enable_enhancement: bool = False, enhancement_strength: int = 2):

        if self.qwen2_vl_model is None or self.qwen2_vl_proc is None:
            return "❌ النموذج غير متاح للاستخدام"

        try:
            # معالجة القيم None
            max_new_tokens = max_new_tokens if max_new_tokens is not None else 600
            temperature = temperature if temperature is not None else 0.3
            top_p = top_p if top_p is not None else 0.9
            top_k = top_k if top_k is not None else 50
            do_sample = do_sample if do_sample is not None else True

            logger.info("✅ معاملات النموذج بعد المعالجة:")
            logger.info(
                f"max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}, top_k={top_k}, do_sample={do_sample}")

            # تحسين الفيديو إذا كان مطلوباً
            final_video_path = video_path
            if enable_enhancement:
                logger.info("🎨 تفعيل تحسين جودة الفيديو...")
                final_video_path = video_enhancer.enhance_video(video_path, enhancement_strength)
                logger.info(f"📹 مسار الفيديو النهائي: {final_video_path}")

            # إعداد الرسائل
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": final_video_path,
                            "max_pixels": pixels_size * pixels_size,
                            "fps": fsp,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # تفريغ الذاكرة
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # معالجة المدخلات
            text = self.qwen2_vl_proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.qwen2_vl_proc(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # نقل المدخلات إلى الجهاز المناسب
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # الاستدلال
            with torch.no_grad():
                generated_ids = self.qwen2_vl_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=self.qwen2_vl_proc.tokenizer.eos_token_id
                )

            # معالجة المخرجات
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            output_text = self.qwen2_vl_proc.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # تنظيف الذاكرة
            del inputs, generated_ids, generated_ids_trimmed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return output_text[0]

        except Exception as e:
            logger.error(f"❌ خطأ في تحليل النشاط: {e}")
            logger.error(traceback.format_exc())
            return f"❌ خطأ في التحليل: {str(e)}"

    def cleanup(self):
        """تنظيف الموارد"""
        try:
            if hasattr(self, 'qwen2_vl_model') and self.qwen2_vl_model is not None:
                del self.qwen2_vl_model
            if hasattr(self, 'qwen2_vl_proc') and self.qwen2_vl_proc is not None:
                del self.qwen2_vl_proc

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("🧹 تم تنظيف موارد ActivityRecognizer")
        except Exception as e:
            logger.error(f"⚠️ خطأ أثناء التنظيف: {e}")
