FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

# استخدام مصادر أسرع وتحديث النظام
RUN sed -i 's/archive.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    libgl1-mesa-glx \
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
    --extra-index-url https://download.pytorch.org/whl/cu124

RUN pip3 install hf_transfer

# نسخ المتطلبات
COPY ./activity_analysis_service/requirements.txt .

RUN pip3 install --timeout 1800 --retries 10 --no-cache-dir \
    opencv-python-headless==4.9.0.80 \
    opencv-contrib-python-headless==4.9.0.80
# تثبيت باقي المتطلبات مع مهلة 30 دقيقة
RUN pip3 install --timeout 1800 --retries 10 --no-cache-dir -r requirements.txt

# نسخ الكود
COPY ./activity_analysis_service/app/activity_model.py .
COPY ./activity_analysis_service/app/activity_service.py .
COPY shared ./shared

# تعيين البيئة
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# فتح المنفذ
EXPOSE 8005

# تشغيل الخدمة
CMD ["uvicorn", "activity_service:app", "--host", "0.0.0.0", "--port", "8005", "--reload"]
