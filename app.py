"""
Веб-интерфейс для демонстрации NER системы
"""

from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import json
import os

app = Flask(__name__, template_folder='templates')  # Указываем папку с шаблонами

# Используем предобученную NER модель
MODEL_NAME = "dslim/bert-base-NER"  # или "dbmdz/bert-large-cased-finetuned-conll03-english"

def load_model():
    """Загрузка предобученной NER модели"""
    try:
        # Загружаем модель и токенизатор
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
        
        # Создаем pipeline для удобства
        nlp_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        
        # Получаем id2tag из модели
        id2tag = model.config.id2label
        
        return model, tokenizer, nlp_pipeline, id2tag
    
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        # Fallback на локальную модель
        try:
            tokenizer = AutoTokenizer.from_pretrained("/app/models/bert-base-uncased")
            # Создаем простую модель (для демонстрации)
            model = AutoModelForTokenClassification.from_pretrained(
                "/app/models/bert-base-uncased",
                num_labels=9
            )
            return model, tokenizer, None, None
        except:
            return None, None, None, None

model, tokenizer, nlp_pipeline, id2tag = load_model()

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
        'model_name': MODEL_NAME,
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
        
        # Если есть pipeline, используем его
        if nlp_pipeline:
            entities = nlp_pipeline(text)
            
            # Форматируем результат
            formatted_entities = []
            for entity in entities:
                formatted_entities.append({
                    'text': entity['word'],
                    'entity': entity['entity_group'],
                    'start': entity['start'],
                    'end': entity['end'],
                    'score': float(entity['score'])
                })
            
            return jsonify({
                'text': text,
                'entities': formatted_entities
            })
        
        # Альтернативный способ без pipeline
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
            outputs = model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
        
        # Обработка результатов
        word_ids = encoding.word_ids()
        previous_word_idx = None
        entities = []
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                tag_id = predictions[i]
                if id2tag:
                    tag = id2tag[tag_id] if tag_id in id2tag else 'O'
                else:
                    tag = f"TAG-{tag_id}"
                
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)