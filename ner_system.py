"""
Многопоточная система NER с правильным сохранением графиков
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
import pandas as pd
from collections import defaultdict
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import random
from tqdm import tqdm
import matplotlib
# Используем агрегированный бэкенд для сохранения без дисплея
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import warnings
import argparse
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time

warnings.filterwarnings('ignore')

# Простая модель BERT для NER
class SimpleBertForNER(nn.Module):
    """Простая модель BERT для NER"""
    
    def __init__(self, model_name, num_tags, dropout_prob=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Вычисляем loss только для неигнорируемых токенов
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.shape[-1])[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss, logits
        
        return logits

# Оптимизированный датасет
class FastNERDataset(Dataset):
    def __init__(self, dataset_split, tokenizer, max_length=128, num_workers=4, debug=False, max_samples=2000):
        self.dataset = dataset_split
        if debug:
            max_samples = min(max_samples, len(self.dataset))
            self.dataset = self.dataset.select(range(max_samples))
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Предобработка с многопоточностью
        print(f"Предобработка {len(self.dataset)} примеров...")
        self.data = self._preprocess_parallel(num_workers)
    
    def _preprocess_parallel(self, num_workers):
        """Многопоточная предобработка данных"""
        results = []
        
        # Используем ThreadPoolExecutor для параллельной обработки
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for idx in range(len(self.dataset)):
                futures.append(executor.submit(self._process_item, idx))
            
            # Собираем результаты
            for future in tqdm(futures, total=len(futures), desc="Предобработка"):
                results.append(future.result())
        
        return results
    
    def _process_item(self, idx):
        """Обработка одного примера"""
        item = self.dataset[idx]
        tokens = item['tokens']
        labels = item['fine_ner_tags']
        
        # Ограничиваем длину последовательности
        tokens = tokens[:self.max_length-2]
        labels = labels[:self.max_length-2]
        
        # Токенизация
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Выравнивание меток
        word_ids = encoding.word_ids()
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                if word_idx < len(labels):
                    label_ids.append(labels[word_idx])
                else:
                    label_ids.append(-100)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Многопоточный тренер
class ParallelTrainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.history = defaultdict(list)
    
    def create_optimizer_scheduler(self, train_loader):
        """Создание оптимизатора и планировщика"""
        
        # Разделяем параметры
        bert_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'bert' in name:
                bert_params.append(param)
            else:
                other_params.append(param)
        
        # Оптимизатор
        optimizer = AdamW([
            {'params': bert_params, 'lr': self.config.learning_rate},
            {'params': other_params, 'lr': self.config.learning_rate * 10}
        ], weight_decay=0.01)
        
        # Планировщик
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, train_loader, optimizer, scheduler, epoch):
        """Обучение одной эпохи"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            start_time = time.time()
            
            # Загрузка данных
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            loss, _ = self.model(input_ids, attention_mask, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            batch_time = time.time() - start_time
            total_loss += loss.item()
            
            # Обновление прогресс-бара
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'batch_time': f'{batch_time:.3f}s'
                })
        
        return total_loss / len(train_loader)
    
    def evaluate(self, loader, id2tag, desc="Оценка"):
        """Оценка модели"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                loss, logits = self.model(input_ids, attention_mask, labels)
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                
                # Обрабатываем батч
                for i in range(len(predictions)):
                    pred_seq = predictions[i].cpu().numpy()
                    true_seq = labels[i].cpu().numpy()
                    mask = attention_mask[i].cpu().numpy()
                    
                    pred_tags = []
                    true_tags = []
                    
                    for j in range(len(pred_seq)):
                        if mask[j] == 1 and true_seq[j] != -100:
                            pred_tags.append(id2tag[pred_seq[j]])
                            true_tags.append(id2tag[true_seq[j]])
                    
                    all_predictions.append(pred_tags)
                    all_true_labels.append(true_tags)
        
        # Вычисляем метрики
        avg_loss = total_loss / len(loader)
        
        try:
            f1 = f1_score(all_true_labels, all_predictions)
            precision = precision_score(all_true_labels, all_predictions)
            recall = recall_score(all_true_labels, all_predictions)
        except:
            f1, precision, recall = 0.0, 0.0, 0.0
        
        return avg_loss, f1, precision, recall, all_predictions, all_true_labels
    
    def train(self, train_loader, val_loader, id2tag, output_dir):
        """Основной цикл обучения"""
        optimizer, scheduler = self.create_optimizer_scheduler(train_loader)
        best_f1 = 0
        
        for epoch in range(self.config.num_epochs):
            print(f"\nЭпоха {epoch+1}/{self.config.num_epochs}")
            print("-" * 40)
            
            # Обучение
            train_loss = self.train_epoch(train_loader, optimizer, scheduler, epoch)
            self.history['train_loss'].append(train_loss)
            
            # Валидация
            val_loss, val_f1, val_precision, val_recall, _, _ = self.evaluate(
                val_loader, id2tag, desc="Валидация"
            )
            
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            self.history['val_precision'].append(val_precision)
            self.history['val_recall'].append(val_recall)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val F1: {val_f1:.4f}")
            print(f"Val Precision: {val_precision:.4f}")
            print(f"Val Recall: {val_recall:.4f}")
            
            # Сохраняем лучшую модель
            if val_f1 > best_f1:
                best_f1 = val_f1
                model_path = os.path.join(output_dir, 'best_model_weights.pth')
                torch.save(self.model.state_dict(), model_path)
                print(f"✓ Сохранена лучшая модель (F1: {val_f1:.4f})")
        
        return self.history

def set_seed(seed):
    """Установка seed для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_training_plots(history, output_dir):
    """Сохранение графиков обучения"""
    if not history or len(history['train_loss']) == 0:
        print("Нет данных для построения графиков")
        return
    
    print("\nСоздание графиков обучения...")
    
    # Создаем фигуру
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # График 1: Loss
    axes[0, 0].plot(history['train_loss'], label='Обучение', marker='o', linewidth=2)
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[0, 0].plot(history['val_loss'], label='Валидация', marker='s', linewidth=2)
    axes[0, 0].set_xlabel('Эпоха', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Функция потерь', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=10)
    
    # График 2: F1-score
    axes[0, 1].plot(history['val_f1'], label='F1-score', color='green', 
                   marker='o', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Эпоха', fontsize=12)
    axes[0, 1].set_ylabel('F1-score', fontsize=12)
    axes[0, 1].set_title('F1-score на валидации', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=10)
    
    # Добавляем значения на точки
    for i, val in enumerate(history['val_f1']):
        axes[0, 1].text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # График 3: Precision и Recall
    axes[1, 0].plot(history['val_precision'], label='Precision', color='orange', 
                   marker='o', linewidth=2, markersize=8)
    axes[1, 0].plot(history['val_recall'], label='Recall', color='red', 
                   marker='s', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Эпоха', fontsize=12)
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].set_title('Precision и Recall', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=10)
    
    # График 4: Сравнение всех метрик
    epochs = range(1, len(history['val_f1']) + 1)
    axes[1, 1].plot(epochs, history['val_f1'], label='F1', color='green', marker='o', linewidth=2)
    axes[1, 1].plot(epochs, history['val_precision'], label='Precision', color='orange', marker='s', linewidth=2)
    axes[1, 1].plot(epochs, history['val_recall'], label='Recall', color='red', marker='^', linewidth=2)
    axes[1, 1].set_xlabel('Эпоха', fontsize=12)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('Сравнение метрик', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=10)
    
    # Настройка layout
    plt.tight_layout(pad=3.0)
    
    # Сохраняем с высоким DPI
    save_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)  # Важно: закрываем фигуру
    
    print(f"✓ Графики сохранены в {save_path}")
    
    # Также сохраняем отдельный график loss
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(history['train_loss'], label='Обучение', marker='o', linewidth=3, markersize=10)
    if 'val_loss' in history and len(history['val_loss']) > 0:
        ax2.plot(history['val_loss'], label='Валидация', marker='s', linewidth=3, markersize=10)
    ax2.set_xlabel('Эпоха', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=14)
    ax2.set_title('Динамика функции потерь', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Добавляем значения на точки
    for i, val in enumerate(history['train_loss']):
        ax2.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    save_path2 = os.path.join(output_dir, 'loss_history.png')
    plt.tight_layout()
    plt.savefig(save_path2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    
    print(f"✓ График loss сохранен в {save_path2}")

def main():
    # Аргументы командной строки
    parser = argparse.ArgumentParser(description='Многопоточная система NER')
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--model', type=str, default="bert-base-uncased")
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_samples', type=int, default=2000)
    parser.add_argument('--max_length', type=int, default=128)
    
    args = parser.parse_args()
    
    # Конфигурация
    class Config:
        def __init__(self, args):
            self.model_name = args.model
            self.max_length = args.max_length
            self.batch_size = args.batch_size
            self.num_epochs = args.epochs
            self.learning_rate = args.lr
            self.dropout_prob = 0.1
            self.seed = 42
            self.warmup_steps = 100
            self.output_dir = "ner_results"
            
            self.num_workers = args.workers
            
            self.debug = args.debug
            self.max_samples = args.max_samples
    
    config = Config(args)
    
    # Настройка
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создание директории для результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("МНОГОПОТОЧНАЯ СИСТЕМА NER ДЛЯ FEW-NERD")
    print("="*60)
    print(f"Устройство: {device}")
    print(f"Модель: {config.model_name}")
    print(f"Воркеров: {config.num_workers}")
    print(f"Размер батча: {config.batch_size}")
    print(f"Эпохи: {config.num_epochs}")
    print(f"Директория результатов: {output_dir}")
    
    # Загрузка данных
    print("\nЗагрузка датасета Few-NERD...")
    try:
        dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")
    except:
        dataset = load_dataset("DFKI-SLT/few-nerd")
    
    # Информация о тегах
    tag_info = dataset['train'].features['fine_ner_tags'].feature.names
    id2tag = {i: tag for i, tag in enumerate(tag_info)}
    tag2id = {tag: i for i, tag in enumerate(tag_info)}
    
    print(f"Количество тегов: {len(tag_info)}")
    print(f"Примеры тегов: {list(tag_info[:10])}")
    
    # Сохранение информации о тегах
    with open(os.path.join(output_dir, 'tag_info.json'), 'w') as f:
        json.dump({'tag2id': tag2id, 'id2tag': id2tag}, f, indent=2)
    
    # Токенизатор
    print("\nЗагрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Датасеты
    print("\nСоздание датасетов...")
    
    train_dataset = FastNERDataset(
        dataset['train'], tokenizer, config.max_length, 
        num_workers=config.num_workers, debug=config.debug, max_samples=config.max_samples
    )
    val_dataset = FastNERDataset(
        dataset['validation'], tokenizer, config.max_length,
        num_workers=config.num_workers, debug=config.debug, max_samples=config.max_samples // 5
    )
    test_dataset = FastNERDataset(
        dataset['test'], tokenizer, config.max_length,
        num_workers=config.num_workers, debug=config.debug, max_samples=config.max_samples // 10
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    print(f"Обучающих батчей: {len(train_loader)}")
    print(f"Валидационных батчей: {len(val_loader)}")
    print(f"Тестовых батчей: {len(test_loader)}")
    
    # Создание модели
    print(f"\nСоздание модели...")
    
    model = SimpleBertForNER(
        config.model_name,
        len(tag_info),
        dropout_prob=config.dropout_prob
    )
    
    model.to(device)
    
    # Информация о модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    
    # Обучение
    print("\n" + "="*50)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("="*50)
    
    trainer = ParallelTrainer(model, config, device)
    start_time = time.time()
    history = trainer.train(train_loader, val_loader, id2tag, output_dir)
    training_time = time.time() - start_time
    
    print(f"\nОбщее время обучения: {training_time:.1f} секунд")
    
    # Тестирование
    print("\n" + "="*50)
    print("ТЕСТИРОВАНИЕ")
    print("="*50)
    
    test_start = time.time()
    test_loss, test_f1, test_precision, test_recall, test_preds, test_true = trainer.evaluate(
        test_loader, id2tag, desc="Тестирование"
    )
    test_time = time.time() - test_start
    
    print(f"\nРезультаты на тестовом наборе:")
    print(f"  Время тестирования: {test_time:.1f} секунд")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  F1-score:  {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    
    # Детальный отчет
    try:
        report = classification_report(test_true, test_preds, digits=4)
        print(f"\nДетальный отчет:\n{report}")
    except:
        report = "Не удалось сгенерировать детальный отчет"
        print("Не удалось сгенерировать детальный отчет")
    
    # Сохранение графиков
    save_training_plots(history, output_dir)
    
    # Сохранение результатов
    print("\n" + "="*50)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*50)
    
    results = {
        'config': {
            'model': config.model_name,
            'epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'workers': config.num_workers,
            'max_length': config.max_length
        },
        'results': {
            'test_f1': float(test_f1),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_loss': float(test_loss),
            'training_time': float(training_time),
            'test_time': float(test_time)
        },
        'history': dict(history)
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Сохраняем отчет
    with open(os.path.join(output_dir, 'test_report.txt'), 'w') as f:
        f.write("МНОГОПОТОЧНАЯ СИСТЕМА NER - РЕЗУЛЬТАТЫ\n")
        f.write("="*60 + "\n\n")
        f.write("РЕЗУЛЬТАТЫ:\n")
        f.write(f"  F1-score:         {test_f1:.4f}\n")
        f.write(f"  Precision:        {test_precision:.4f}\n")
        f.write(f"  Recall:           {test_recall:.4f}\n")
        f.write(f"  Loss:             {test_loss:.4f}\n\n")
        f.write("ДЕТАЛЬНЫЙ ОТЧЕТ:\n")
        f.write(report)
    
    # Сохраняем модель
    model_save_path = os.path.join(output_dir, 'model')
    os.makedirs(model_save_path, exist_ok=True)
    
    # Сохраняем веса модели
    torch.save(model.state_dict(), os.path.join(model_save_path, 'model_weights.pth'))
    
    # Сохраняем конфигурацию модели
    model_config = {
        'model_name': config.model_name,
        'num_tags': len(tag_info),
        'dropout_prob': config.dropout_prob
    }
    
    with open(os.path.join(model_save_path, 'config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Сохраняем токенизатор
    tokenizer.save_pretrained(model_save_path)
    
    # Сохраняем информацию о тегах
    with open(os.path.join(model_save_path, 'tags.json'), 'w') as f:
        json.dump({'id2tag': id2tag, 'tag2id': tag2id}, f, indent=2)
    
    print(f"\n✓ Все результаты сохранены в: {output_dir}")
    print(f"✓ Модель сохранена в: {model_save_path}")
    
    # Создаем HTML отчет с графиками
    create_html_report(output_dir, results, history)
    
    print("\n✅ Обучение завершено успешно!")

def create_html_report(output_dir, results, history):
    """Создание HTML отчета с графиками"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Отчет по обучению NER модели</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #555; margin-top: 30px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .card {{ background: #f9f9f9; padding: 20px; margin: 20px 0; border-radius: 10px; }}
            .results {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
            .result-item {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ color: #7f8c8d; font-size: 14px; }}
            .images {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 30px; }}
            img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Отчет по обучению NER модели</h1>
            <p>Дата: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="card">
                <h2>Конфигурация</h2>
                <ul>
                    <li><strong>Модель:</strong> {results['config']['model']}</li>
                    <li><strong>Эпохи:</strong> {results['config']['epochs']}</li>
                    <li><strong>Размер батча:</strong> {results['config']['batch_size']}</li>
                    <li><strong>Learning rate:</strong> {results['config']['learning_rate']}</li>
                    <li><strong>Workers:</strong> {results['config']['workers']}</li>
                </ul>
            </div>
            
            <div class="card">
                <h2>Результаты</h2>
                <div class="results">
                    <div class="result-item">
                        <div class="metric">{results['results']['test_f1']:.4f}</div>
                        <div class="metric-label">F1-Score</div>
                    </div>
                    <div class="result-item">
                        <div class="metric">{results['results']['test_precision']:.4f}</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    <div class="result-item">
                        <div class="metric">{results['results']['test_recall']:.4f}</div>
                        <div class="metric-label">Recall</div>
                    </div>
                    <div class="result-item">
                        <div class="metric">{results['results']['test_loss']:.4f}</div>
                        <div class="metric-label">Loss</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Графики обучения</h2>
                <div class="images">
                    <div>
                        <h3>Динамика обучения</h3>
                        <img src="training_history.png" alt="Training History">
                    </div>
                    <div>
                        <h3>Функция потерь</h3>
                        <img src="loss_history.png" alt="Loss History">
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Производительность</h2>
                <ul>
                    <li><strong>Время обучения:</strong> {results['results']['training_time']:.1f} секунд</li>
                    <li><strong>Время тестирования:</strong> {results['results']['test_time']:.1f} секунд</li>
                    <li><strong>Общее время:</strong> {results['results']['training_time'] + results['results']['test_time']:.1f} секунд</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'report.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ HTML отчет создан: {os.path.join(output_dir, 'report.html')}")

if __name__ == "__main__":
    main()