#!/bin/bash

# Скрипт запуска Docker контейнера

set -e

echo "=========================================="
echo "Starting NER System Docker Container"
echo "=========================================="

# Проверяем наличие GPU
if [ "$USE_GPU" = "true" ]; then
    echo "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected"
        echo "GPU Info:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    else
        echo "⚠️  NVIDIA GPU not detected, using CPU"
    fi
fi

# Создаем директории если их нет
mkdir -p /app/data
mkdir -p /app/results
mkdir -p /app/models

# Проверяем наличие предобученной модели
if [ -f "/app/models/model_weights.pth" ]; then
    echo "✅ Pre-trained model found"
else
    echo "📥 No pre-trained model found, will train from scratch"
fi

# Проверяем наличие датасета
if [ "$DOWNLOAD_DATA" = "true" ]; then
    echo "📥 Downloading dataset..."
    python -c "from datasets import load_dataset; load_dataset('DFKI-SLT/few-nerd', 'supervised')"
fi

# Запускаем в зависимости от режима
if [ "$MODE" = "train" ]; then
    echo "🚀 Starting training..."
    python run.py \
        --mode train \
        --epochs ${EPOCHS:-3} \
        --batch_size ${BATCH_SIZE:-32} \
        --workers ${WORKERS:-4} \
        --model ${MODEL_NAME:-bert-base-uncased} \
        --lr ${LEARNING_RATE:-2e-5} \
        --max_samples ${MAX_SAMPLES:-5000}
        
elif [ "$MODE" = "api" ]; then
    echo "🌐 Starting API server..."
    python app.py
    
elif [ "$MODE" = "jupyter" ]; then
    echo "📓 Starting Jupyter Notebook..."
    jupyter notebook \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --NotebookApp.token='' \
        --NotebookApp.password=''
        
elif [ "$MODE" = "test" ]; then
    echo "🧪 Running tests..."
    python run.py --mode test
    
else
    echo "❓ Unknown mode: $MODE"
    echo "Available modes: train, api, jupyter, test"
    exit 1
fi