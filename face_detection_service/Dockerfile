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
COPY requirements.txt .

# تثبيت Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
RUN pip3 install -r requirements.txt

# نسخ الكود وملفات النموذج
COPY faces_model.py .
COPY faces_service.py .
COPY shared ./shared
COPY scrfd.onnx .  # إذا كان الملف موجوداً

# تعيين البيئة
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# فتح المنفذ
EXPOSE 8002

# تشغيل الخدمة
CMD ["uvicorn", "faces_service:app", "--host", "127.0.0.1", "--port", "8002"]