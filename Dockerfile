FROM nvidia/cuda:12.4.0-runtime-ubuntu20.04

WORKDIR /app

# استخدام مصادر أسرع وتحديث النظام
RUN sed -i 's/archive.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# استخدام pip مع cache وتثبيت PyTorch أولاً
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir \
    torch==2.4.1+cu118 \
    torchvision==0.19.1+cu118 \
    torchaudio==2.4.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# نسخ المتطلبات أولاً (لتحسين caching)
COPY requirements.txt .

# تثبيت باقي المتطلبات
RUN pip3 install --no-cache-dir -r requirements.txt

# نسخ الكود
COPY . .

# تعيين البيئة
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "audio_service:app", "--host", "127.0.0.1", "--port", "8000"]