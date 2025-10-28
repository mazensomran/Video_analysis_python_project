FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

# Ubuntu 22.04 يأتي مع Python 3.10 افتراضيًا - لا حاجة لتثبيت إضافي
RUN sed -i 's/archive.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean



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

# نسخ المتطلبات أولاً (لتحسين caching)
COPY requirements.txt .

RUN pip3 install --timeout 1800 --retries 10 --no-cache-dir \
    opencv-python-headless==4.9.0.80 \
    opencv-contrib-python-headless==4.9.0.80
# تثبيت باقي المتطلبات مع مهلة 30 دقيقة
RUN pip3 install --timeout 1800 --retries 10 --no-cache-dir -r requirements.txt

# نسخ الكود
COPY main.py .
COPY shared ./shared

# إنشاء المجلدات إذا لم تكن موجودة
RUN mkdir -p uploads outputs

# تعيين البيئة
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--workers", "1"]
