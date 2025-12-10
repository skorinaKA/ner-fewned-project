#!/bin/bash

# Скрипт для загрузки предобученных моделей

echo "Downloading pre-trained models..."

MODELS_DIR="/app/models"
mkdir -p $MODELS_DIR

# Загружаем BERT
echo "Downloading BERT models..."
python -c "
from transformers import AutoTokenizer, AutoModel
import torch

models = [
    'bert-base-uncased',
    'distilbert-base-uncased',
    'roberta-base'
]

for model_name in models:
    print(f'Downloading {model_name}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Сохраняем
    save_path = f'$MODELS_DIR/{model_name}'
    tokenizer.save_pretrained(save_path)
    torch.save(model.state_dict(), f'{save_path}/pytorch_model.bin')
    print(f'Saved to {save_path}')
"

echo "✅ Models downloaded to $MODELS_DIR"