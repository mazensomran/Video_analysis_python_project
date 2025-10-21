import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any
import logging
from scrfd import SCRFD, Threshold

logger = logging.getLogger(__name__)


class FaceDetector:
    def __init__(self, face_threshold=0.3):
        try:
            self.scrfd_detector = SCRFD.from_path("../scrfd.onnx",
                                                  providers=["CUDAExecutionProvider"])
            if self.scrfd_detector is None:
                raise Exception("فشل تحميل نموذج SCRFD لكشف الوجوه.")

            self.threshold = Threshold(probability=face_threshold)
            self.frame_counter = 0
            self.nms_threshold = 0.4

            print("✅ تم تهيئة FaceDetector بنجاح")

        except Exception as e:
            print(f"❌ خطأ في تهيئة FaceDetector: {e}")
            self.scrfd_detector = None

    def detect_faces(self, frame: np.ndarray,threshold:float) -> List[Dict[str, Any]]:
        """كشف الوجوه في إطار"""
        if self.scrfd_detector is None:
            return []

        self.frame_counter += 1

        height, width, _ = frame.shape
        all_raw_detections = []
        processed_frame = frame.copy()

        try:
            # تحويل الإطار إلى PIL Image
            pil_frame = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

            # الكشف عن الوجوه
            faces_in_frame = self.scrfd_detector.detect(pil_frame, threshold=self.threshold)

            # جمع الاكتشافات
            for face in faces_in_frame:
                x1 = int(face.bbox.upper_left.x)
                y1 = int(face.bbox.upper_left.y)
                x2 = int(face.bbox.lower_right.x)
                y2 = int(face.bbox.lower_right.y)

                # التأكد من أن الإحداثيات ضمن حدود الإطار
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(x1, min(x2, width))
                y2 = max(y1, min(y2, height))

                all_raw_detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": face.probability,
                    "keypoints": face.keypoints
                })

            # تطبيق NMS
            final_detections = self._apply_nms(all_raw_detections)

            faces = []
            for i, det in enumerate(final_detections):
                if det["confidence"] >= threshold:
                    x1, y1, x2, y2 = det["bbox"]
                    width_face = x2 - x1
                    height_face = y2 - y1

                    faces.append({
                        "bbox": [x1, y1, width_face, height_face],
                        "confidence": det["confidence"],
                        "face_id": i,
                        "keypoints": {
                            "left_eye": {"x": det["keypoints"].left_eye.x, "y": det["keypoints"].left_eye.y},
                            "right_eye": {"x": det["keypoints"].right_eye.x, "y": det["keypoints"].right_eye.y},
                            "nose": {"x": det["keypoints"].nose.x, "y": det["keypoints"].nose.y},
                            "left_mouth": {"x": det["keypoints"].left_mouth.x, "y": det["keypoints"].left_mouth.y},
                            "right_mouth": {"x": det["keypoints"].right_mouth.x, "y": det["keypoints"].right_mouth.y}
                        }
                    })

            return faces

        except Exception as e:
            logger.error(f"❌ خطأ في كشف الوجوه: {e}")
            return []

    def _apply_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """تطبيق Non-Maximum Suppression على الاكتشافات"""
        if not detections:
            return []

        boxes = np.array([d["bbox"] for d in detections])
        scores = np.array([d["confidence"] for d in detections])

        boxes_xywh = np.array([[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes])

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            scores.tolist(),
            score_threshold=float(self.threshold.probability),
            nms_threshold=self.nms_threshold
        )

        if isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()
        elif isinstance(indices, list) and len(indices) > 0 and isinstance(indices[0], list):
            indices = [i[0] for i in indices]
        else:
            indices = []

        return [detections[i] for i in indices]

    def cleanup(self):
        """تنظيف الموارد"""
        if hasattr(self, 'scrfd_detector'):
            self.scrfd_detector = None
            logger.info("🧹 تم تنظيف موارد FaceDetector")
