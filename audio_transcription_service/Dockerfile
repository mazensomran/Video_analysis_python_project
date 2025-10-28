FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

# استخدام مصادر أسرع وتحديث النظام
RUN sed -i 's/archive.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# التحقق من إصدار Python (سيكون 3.10 افتراضيًا في Ubuntu 22.04)
RUN python3 --version

# تكوين pip للمهلة الطويلة والمرايا السريعة
RUN pip3 install --upgrade pip && \
    pip3 config set global.timeout 1800 && \
    pip3 config set global.retries 15 && \
    pip3 config set global.trusted-host "pypi.org" && \
    pip3 config set global.trusted-host "pypi.python.org" && \
    pip3 config set global.trusted-host "files.pythonhosted.org" && \
    pip3 config set global.trusted-host "download.pytorch.org"

# استخدام مرايا Tsinghua لـ PyPI (أسرع بكثير)
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# تثبيت PyTorch مع مهلة 30 دقيقة
RUN pip3 install --timeout 1800 --retries 10 --no-cache-dir \
    torch==2.4.1+cu124 \
    torchvision==0.19.1+cu124 \
    torchaudio==2.4.1+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# نسخ المتطلبات
COPY requirements.txt .

# تثبيت باقي المتطلبات مع مهلة 30 دقيقة
RUN pip3 install --timeout 1800 --retries 10 --no-cache-dir -r requirements.txt

# نسخ الكود
COPY ./audio_transcription_service/app/audio_model.py .
COPY ./audio_transcription_service/app/audio_service.py .
COPY shared ./shared

# تعيين البيئة
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# فتح المنفذ
EXPOSE 8001

# تشغيل الخدمة - تصحيح من 127.0.0.1 إلى 0.0.0.0
CMD ["uvicorn", "audio_service:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
