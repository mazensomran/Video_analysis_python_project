FROM nvidia/cuda:12.4.0-runtime-ubuntu20.04

WORKDIR /app

# تثبيت dependencies النظام
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# نسخ المتطلبات
COPY requirements.txt .

# تثبيت Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install torch==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
RUN pip3 install -r requirements.txt

# نسخ الكود
COPY texts_model.py .
COPY texts_service.py .
COPY shared ./shared

# تعيين البيئة
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# فتح المنفذ
EXPOSE 8003

# تشغيل الخدمة
CMD ["uvicorn", "texts_service:app", "--host", "127.0.0.1", "--port", "8003"]