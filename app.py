"""
Веб-интерфейс для демонстрации NER системы
"""

from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer
import json
import os
from ner_system import SimpleBertForNER
import numpy as np

app = Flask(__name__)

# Загрузка модели
MODEL_PATH = "/app/models/best_model"
TAGS_PATH = "/app/models/tags.json"

def load_model():
    """Загрузка модели и токенизатора"""
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    
    try:
        # Загружаем информацию о тегах
        with open(TAGS_PATH, 'r') as f:
            tag_info = json.load(f)
            id2tag = tag_info['id2tag']
            tag2id = tag_info['tag2id']
        
        # Загружаем токенизатор
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # Загружаем модель
        model = SimpleBertForNER(
            model_name="bert-base-uncased",
            num_tags=len(id2tag)
        )
        
        # Загружаем веса
        weights_path = os.path.join(MODEL_PATH, "model_weights.pth")
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        
        model.eval()
        
        return model, tokenizer, id2tag
    
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None, None, None

model, tokenizer, id2tag = load_model()

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Проверка здоровья сервиса"""
    status = {
        'status': 'healthy' if model is not None else 'no_model',
        'model_loaded': model is not None,
        'gpu_available': torch.cuda.is_available()
    }
    return jsonify(status)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API для предсказания сущностей"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        # Токенизация
        tokens = text.split()
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Предсказание
        with torch.no_grad():
            logits = model(encoding['input_ids'], encoding['attention_mask'])
            predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
        
        # Обработка результатов
        word_ids = encoding.word_ids()
        previous_word_idx = None
        entities = []
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                tag_id = predictions[i]
                tag = id2tag[str(tag_id)] if str(tag_id) in id2tag else 'O'
                if tag != 'O':
                    entities.append({
                        'word': tokens[word_idx],
                        'entity': tag,
                        'position': word_idx
                    })
            previous_word_idx = word_idx
        
        # Группируем последовательные сущности
        merged_entities = []
        current_entity = None
        
        for entity in entities:
            if (current_entity and 
                entity['entity'] == current_entity['entity'] and
                entity['position'] == current_entity['end_position'] + 1):
                # Продолжаем ту же сущность
                current_entity['text'] += ' ' + entity['word']
                current_entity['end_position'] = entity['position']
            else:
                # Новая сущность
                if current_entity:
                    merged_entities.append(current_entity)
                current_entity = {
                    'text': entity['word'],
                    'entity': entity['entity'],
                    'start_position': entity['position'],
                    'end_position': entity['position']
                }
        
        if current_entity:
            merged_entities.append(current_entity)
        
        return jsonify({
            'text': text,
            'entities': merged_entities,
            'tokens': tokens
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """API для запуска обучения"""
    # В реальном приложении здесь будет асинхронный запуск обучения
    return jsonify({
        'message': 'Training started',
        'status': 'processing'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)