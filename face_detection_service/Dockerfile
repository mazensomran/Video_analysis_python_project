FROM nvidia/cuda:12.4.0-runtime-ubuntu20.04

WORKDIR /app

# تثبيت dependencies النظام
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# نسخ المتطلبات
COPY ./face_detection_service/requirements.txt .

# تثبيت Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir \
    torch==2.4.1+cu118 \
    torchvision==0.19.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118 \

RUN pip3 install -r requirements.txt

# نسخ الكود وملفات النموذج
COPY ./face_detection_service/app/faces_model.py .
COPY ./face_detection_service/app/faces_service.py .
COPY shared ./shared
COPY ./face_detection_service/scrfd.onnx .

# تعيين البيئة
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# فتح المنفذ
EXPOSE 8002

# تشغيل الخدمة
CMD ["uvicorn", "faces_service:app", "--host", "127.0.0.1", "--port", "8002"]