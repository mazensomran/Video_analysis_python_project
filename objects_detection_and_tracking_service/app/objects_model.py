import torch
import cv2
import numpy as np
import logging
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
import supervision as sv

logger = logging.getLogger(__name__)


class ObjectTracker:
    def __init__(self, object_threshold=0.5):
        self.trajectories = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.image_processor = None
        self.id2label = None
        self.frame_counter = 0
        self.tracker = None
        self.box_annotator = None
        self.label_annotator = None
        self.trace_annotator = None
        self.detection_threshold = object_threshold

        self._load_model()

    def _load_model(self):
        """تحميل نموذج RT-DETR"""
        try:
            logger.info(f"📥 تحميل نموذج RT-DETR v2 r18vd على {self.device}...")
            self.image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
            self.model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd").to(self.device)
            self.model.eval()

            self.id2label = self.model.config.id2label

            if 0 not in self.id2label or self.id2label[0] != "person":
                logger.warning("⚠️ فئة 'person' (ID 0) غير موجودة في id2label.")

            logger.info(f"✅ تم تحميل RT-DETR v2 بنجاح. عدد الفئات: {len(self.id2label)}")

        except Exception as e:
            logger.error(f"❌ خطأ في تحميل نموذج RT-DETR v2: {e}")
            raise Exception(f"فشل تحميل نموذج RT-DETR v2 r18vd: {e}")

        # تهيئة مكونات supervision
        try:
            self.tracker = sv.ByteTrack()
            self.box_annotator = sv.BoxAnnotator()
            self.label_annotator = sv.LabelAnnotator()
            self.trace_annotator = sv.TraceAnnotator()
            self.frame_counter = 0

            logger.info(f"✅ تم تهيئة ObjectTracker (ByteTrack + RT-DETR v2) بنجاح")
        except Exception as e:
            logger.error(f"❌ خطأ في تهيئة مكونات supervision: {e}")
            raise Exception(f"فشل في تهيئة ObjectTracker: {e}")

    def track_objects(self, frame: np.ndarray,threshold: float):
        """يكشف ويتتبع جميع الكائنات"""
        all_detections = []

        try:
            if self.model is None or self.image_processor is None:
                logger.error("❌ النموذج غير مُحمل.")
                return sv.Detections.empty(), []

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = self.image_processor(images=rgb_frame, return_tensors="pt").to(self.device)

            with torch.no_grad():
                with torch.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu",
                                    dtype=torch.float16 if self.device.type == "cuda" else torch.float32):
                    outputs = self.model(**inputs)

            processed_results = self.image_processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor([(h, w)], device=self.device),
                threshold=self.detection_threshold
            )[0]

            if 'boxes' not in processed_results or len(processed_results['boxes']) == 0:
                return sv.Detections.empty(), []

            detections = sv.Detections.from_transformers(
                transformers_results=processed_results,
                id2label=self.id2label
            )

            if detections.confidence is not None:
                mask = detections.confidence > self.detection_threshold
                detections = detections[mask]

            if len(detections) == 0:
                return sv.Detections.empty(), []

            detections = self.tracker.update_with_detections(detections)

            for i in range(len(detections)):
                if detections.xyxy is None or i >= len(detections.xyxy):
                    continue

                bbox = detections.xyxy[i].tolist()
                if len(bbox) != 4:
                    continue

                conf = detections.confidence[i] if detections.confidence is not None else 0.0
                cls_id = detections.class_id[i] if detections.class_id is not None else 0
                x1, y1, x2, y2 = map(int, bbox)
                class_name = self.id2label.get(int(cls_id), f"unknown_{cls_id}")
                track_id = detections.tracker_id[i] if detections.tracker_id is not None else None

                all_detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_name": class_name,
                        "track_id": track_id
                })

            return all_detections

        except Exception as e:
            logger.error(f"❌ خطأ في كشف وتتبع الكائنات: {e}")
            return sv.Detections.empty(), []

    def cleanup(self):
        """تنظيف الموارد"""
        try:
            self.model = None
            self.image_processor = None
            self.tracker = None
            self.box_annotator = None
            self.label_annotator = None
            self.trace_annotator = None
            self.id2label = None
            logger.info("🧹 تم تنظيف موارد ObjectTracker")
        except Exception as e:
            logger.error(f"⚠️ خطأ في تنظيف ObjectTracker: {e}")