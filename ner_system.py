"""
Многопоточная система NER с ускорением вычислений
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler  # Для смешанной точности
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
import warnings
import argparse
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from functools import lru_cache
import hashlib

warnings.filterwarnings('ignore')

# Настройка многопоточности для PyTorch
torch.set_num_threads(4)  # Для CPU операций
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Автоматическая оптимизация для CUDA
    torch.backends.cudnn.enabled = True

# Исправленный CRF слой с поддержкой многопоточности
class ParallelCRF(nn.Module):
    """Многопоточный CRF слой с батч-оптимизациями"""
    
    def __init__(self, num_tags, batch_first=True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # Матрица переходов
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        # Кеш для часто используемых вычислений
        self._transitions_t = None
        self._start_transitions_expanded = None
        
    def _precompute_matrices(self, batch_size, device):
        """Предвычисление матриц для ускорения"""
        if self._transitions_t is None:
            self._transitions_t = self.transitions.T.unsqueeze(0)  # (1, num_tags, num_tags)
        if self._start_transitions_expanded is None:
            self._start_transitions_expanded = self.start_transitions.unsqueeze(0)  # (1, num_tags)
    
    def forward(self, emissions, tags, mask=None, reduction='mean'):
        batch_size, seq_len, num_tags = emissions.shape
        
        if mask is None:
            mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=emissions.device)
        
        # Предвычисление
        self._precompute_matrices(batch_size, emissions.device)
        
        # Вычисление логарифмического правдоподобия
        numerator = self._compute_score_parallel(emissions, tags, mask)
        denominator = self._compute_normalizer_parallel(emissions, mask)
        llh = (numerator - denominator) * mask.sum(dim=1)
        
        if reduction == 'mean':
            return -llh.mean()
        elif reduction == 'sum':
            return -llh.sum()
        else:
            return -llh
    
    def _compute_score_parallel(self, emissions, tags, mask):
        """Параллельное вычисление скора"""
        batch_size, seq_len = tags.shape
        
        # Используем tensor operations вместо циклов
        score = self.start_transitions_expanded[0, tags[:, 0]]
        score += emissions[torch.arange(batch_size), 0, tags[:, 0]]
        
        # Векторизованные вычисления для всего батча
        for i in range(1, seq_len):
            # Получаем переходы в виде матрицы
            prev_tags = tags[:, i-1]
            curr_tags = tags[:, i]
            
            # Векторизованное вычисление переходов
            transition_scores = self.transitions[curr_tags, prev_tags]
            emission_scores = emissions[torch.arange(batch_size), i, curr_tags]
            
            # Применяем маску
            mask_i = mask[:, i]
            score += (transition_scores + emission_scores) * mask_i
        
        # Добавляем конечные переходы
        last_idx = mask.sum(dim=1).long() - 1
        last_tags = tags[torch.arange(batch_size), last_idx]
        score += self.end_transitions[last_tags]
        
        return score
    
    def _compute_normalizer_parallel(self, emissions, mask):
        """Параллельный алгоритм forward"""
        batch_size, seq_len, num_tags = emissions.shape
        
        # Инициализация с предвычисленными start transitions
        alpha = self.start_transitions_expanded + emissions[:, 0]
        
        for i in range(1, seq_len):
            # Векторизованные вычисления
            emissions_i = emissions[:, i].unsqueeze(1)  # (batch, 1, num_tags)
            alpha_expanded = alpha.unsqueeze(2)  # (batch, num_tags, 1)
            
            # Используем предвычисленную матрицу переходов
            scores = alpha_expanded + emissions_i + self._transitions_t
            
            # Log sum exp по измерениям
            alpha_new = torch.logsumexp(scores, dim=1)
            
            # Применяем маску
            mask_i = mask[:, i].unsqueeze(1)
            alpha = alpha_new * mask_i + alpha * (1 - mask_i)
        
        # Добавляем end transitions
        alpha += self.end_transitions.unsqueeze(0)
        
        # Log sum exp по всем тегам
        log_z = torch.logsumexp(alpha, dim=1)
        
        return log_z.sum()
    
    def decode(self, emissions, mask=None):
        """Параллельное декодирование Витерби"""
        batch_size, seq_len, num_tags = emissions.shape
        
        if mask is None:
            mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=emissions.device)
        
        return self._viterbi_decode_parallel(emissions, mask)
    
    def _viterbi_decode_parallel(self, emissions, mask):
        """Параллельный алгоритм Витерби"""
        batch_size, seq_len, num_tags = emissions.shape
        
        # Инициализация
        viterbi_score = self.start_transitions_expanded + emissions[:, 0]
        backpointers = torch.zeros((batch_size, seq_len, num_tags), 
                                  dtype=torch.long, device=emissions.device)
        
        # Проход вперед
        for i in range(1, seq_len):
            emissions_i = emissions[:, i].unsqueeze(1)
            scores_expanded = viterbi_score.unsqueeze(2)
            
            # Векторизованное вычисление скоров
            scores = scores_expanded + emissions_i + self._transitions_t
            
            # Поиск максимумов
            max_scores, max_indices = scores.max(dim=1)
            backpointers[:, i] = max_indices
            
            # Обновление с учетом маски
            mask_i = mask[:, i].unsqueeze(1)
            viterbi_score = max_scores * mask_i + viterbi_score * (1 - mask_i)
        
        # Добавляем end transitions
        viterbi_score += self.end_transitions.unsqueeze(0)
        
        # Находим лучшие теги
        _, best_tags = viterbi_score.max(dim=1)
        
        # Обратный проход
        best_paths = torch.zeros((batch_size, seq_len), dtype=torch.long, device=emissions.device)
        best_paths[:, -1] = best_tags
        
        for i in range(seq_len - 2, -1, -1):
            batch_indices = torch.arange(batch_size, device=emissions.device)
            best_tags = backpointers[batch_indices, i + 1, best_tags]
            best_paths[:, i] = best_tags
        
        return best_paths
    
    @property
    def start_transitions_expanded(self):
        if self._start_transitions_expanded is None:
            self._start_transitions_expanded = self.start_transitions.unsqueeze(0)
        return self._start_transitions_expanded

# Кеширующий датасет с предварительной обработкой
class CachedNERDataset(Dataset):
    """Датасет с кешированием предобработанных данных"""
    
    def __init__(self, dataset_split, tokenizer, max_length=128, cache_dir="./cache", 
                 num_workers=4, debug=False):
        self.dataset = dataset_split
        if debug:
            self.dataset = self.dataset.select(range(min(2000, len(self.dataset))))
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.num_workers = num_workers
        
        # Создаем кеш-директорию
        os.makedirs(cache_dir, exist_ok=True)
        
        # Предварительная обработка с многопоточностью
        self._preprocess_parallel()
    
    def _preprocess_parallel(self):
        """Многопоточная предобработка данных"""
        print(f"Предобработка {len(self.dataset)} примеров с {self.num_workers} потоками...")
        
        # Создаем пул потоков
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Подготавливаем аргументы
            args = [(i, item) for i, item in enumerate(self.dataset)]
            
            # Обрабатываем параллельно
            results = list(tqdm(
                executor.map(self._process_item, args),
                total=len(args),
                desc="Предобработка"
            ))
        
        self.cached_data = results
    
    def _process_item(self, args):
        """Обработка одного элемента"""
        idx, item = args
        
        # Генерируем уникальный ключ для кеша
        cache_key = hashlib.md5(
            f"{idx}_{''.join(item['tokens'][:10])}".encode()
        ).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pt")
        
        # Пробуем загрузить из кеша
        if os.path.exists(cache_path):
            try:
                return torch.load(cache_path)
            except:
                pass
        
        # Если нет в кеше, обрабатываем
        tokens = item['tokens']
        labels = item['fine_ner_tags']
        
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
                label_ids.append(labels[word_idx] if word_idx < len(labels) else -100)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        # Создаем результат
        result = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }
        
        # Сохраняем в кеш
        torch.save(result, cache_path)
        
        return result
    
    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, idx):
        return self.cached_data[idx]

# Модель с поддержкой смешанной точности
class FastBertCRFForNER(nn.Module):
    """Оптимизированная модель с поддержкой смешанной точности"""
    
    def __init__(self, model_name, num_tags, dropout_prob=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)
        self.crf = ParallelCRF(num_tags, batch_first=True)
        
        # Инициализация весов классификатора
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Получаем эмбеддинги BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        
        # Применяем dropout и классификатор
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        
        if labels is not None:
            # Создаем маску для валидных токенов
            mask = (labels != -100) & (attention_mask == 1)
            
            # Заменяем -100 на 0 для CRF
            labels_for_crf = labels.clone()
            labels_for_crf[labels == -100] = 0
            
            # Вычисляем CRF loss
            loss = self.crf(emissions, labels_for_crf, mask=mask, reduction='mean')
            return loss, emissions
        else:
            # Для инференса
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return predictions

# Многопоточный тренер со смешанной точностью
class ParallelNERTrainer:
    """Тренер с поддержкой многопоточности и смешанной точности"""
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        # Для смешанной точности
        self.scaler = GradScaler() if device.type == 'cuda' else None
        
        # История обучения
        self.history = defaultdict(list)
        
        # Статистика времени
        self.timing_stats = defaultdict(list)
    
    def create_optimizer_scheduler(self, train_loader):
        """Создание оптимизатора и планировщика с разными LR"""
        
        # Разделяем параметры
        bert_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'bert' in name:
                bert_params.append(param)
            else:
                other_params.append(param)
        
        # Оптимизатор с разными learning rates
        optimizer = torch.optim.AdamW([
            {'params': bert_params, 'lr': self.config.LEARNING_RATE},
            {'params': other_params, 'lr': self.config.LEARNING_RATE * 10}
        ], weight_decay=self.config.WEIGHT_DECAY)
        
        # Планировщик
        total_steps = len(train_loader) * self.config.NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler
    
    def train_epoch_parallel(self, train_loader, optimizer, scheduler, epoch):
        """Многопоточная эпоха обучения со смешанной точностью"""
        self.model.train()
        total_loss = 0
        batch_times = []
        
        progress_bar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            start_time = time.time()
            
            # Перемещаем данные на устройство
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Обнуляем градиенты
            optimizer.zero_grad()
            
            # Forward pass со смешанной точностью
            if self.scaler is not None and self.device.type == 'cuda':
                with autocast():
                    loss, _ = self.model(input_ids, attention_mask, labels)
                
                # Backward pass с масштабированием градиентов
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                
                # Обрезание градиентов
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.MAX_GRAD_NORM
                )
                
                # Шаг оптимизатора
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Без смешанной точности
                loss, _ = self.model(input_ids, attention_mask, labels)
                loss.backward()
                
                # Обрезание градиентов
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.MAX_GRAD_NORM
                )
                
                # Шаг оптимизатора
                optimizer.step()
            
            # Шаг планировщика
            scheduler.step()
            
            # Логирование
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            total_loss += loss.item()
            
            # Обновление прогресс-бара
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'batch_time': f'{batch_time:.3f}s',
                    'lr': scheduler.get_last_lr()[0]
                })
        
        # Статистика времени
        avg_batch_time = np.mean(batch_times)
        self.timing_stats['batch_times'].extend(batch_times)
        
        print(f"Среднее время батча: {avg_batch_time:.3f}s")
        print(f"Примеров в секунду: {self.config.BATCH_SIZE / avg_batch_time:.1f}")
        
        return total_loss / len(train_loader)
    
    def evaluate_parallel(self, loader, id2tag):
        """Параллельная оценка модели"""
        self.model.eval()
        total_loss = 0
        
        # Используем ProcessPoolExecutor для параллельного вычисления метрик
        all_predictions = []
        all_true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Оценка", leave=False):
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # Forward pass
                if self.scaler is not None and self.device.type == 'cuda':
                    with autocast():
                        loss, emissions = self.model(input_ids, attention_mask, labels)
                else:
                    loss, emissions = self.model(input_ids, attention_mask, labels)
                
                total_loss += loss.item()
                
                # Декодирование
                predictions = self.model.crf.decode(emissions, mask=attention_mask.bool())
                
                # Собираем предсказания и истинные метки для параллельной обработки
                batch_data = []
                for i in range(len(predictions)):
                    batch_data.append({
                        'pred': predictions[i].cpu().numpy(),
                        'true': labels[i].cpu().numpy(),
                        'mask': attention_mask[i].cpu().numpy(),
                        'id2tag': id2tag
                    })
                
                # Параллельная обработка батча
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(self._process_prediction, batch_data))
                
                for pred_tags, true_tags in results:
                    all_predictions.append(pred_tags)
                    all_true_labels.append(true_tags)
        
        # Вычисляем метрики
        avg_loss = total_loss / len(loader)
        
        # Параллельное вычисление метрик
        with ThreadPoolExecutor(max_workers=3) as executor:
            f1_future = executor.submit(f1_score, all_true_labels, all_predictions)
            precision_future = executor.submit(precision_score, all_true_labels, all_predictions)
            recall_future = executor.submit(recall_score, all_true_labels, all_predictions)
            
            f1 = f1_future.result()
            precision = precision_future.result()
            recall = recall_future.result()
        
        return avg_loss, f1, precision, recall, all_predictions, all_true_labels
    
    def _process_prediction(self, data):
        """Обработка одного предсказания"""
        pred_tags = []
        true_tags = []
        
        for j in range(len(data['pred'])):
            if data['mask'][j] == 1 and data['true'][j] != -100:
                pred_tags.append(data['id2tag'][data['pred'][j]])
                true_tags.append(data['id2tag'][data['true'][j]])
        
        return pred_tags, true_tags
    
    def train(self, train_loader, val_loader, id2tag, output_dir):
        """Основной цикл обучения"""
        print("\n" + "="*60)
        print("МНОГОПОТОЧНОЕ ОБУЧЕНИЕ")
        print("="*60)
        
        # Оптимизатор и планировщик
        optimizer, scheduler = self.create_optimizer_scheduler(train_loader)
        
        best_f1 = 0
        best_model_path = os.path.join(output_dir, self.config.MODEL_SAVE_PATH)
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nЭпоха {epoch+1}/{self.config.NUM_EPOCHS}")
            print("-" * 40)
            
            # Обучение
            epoch_start = time.time()
            train_loss = self.train_epoch_parallel(train_loader, optimizer, scheduler, epoch)
            train_time = time.time() - epoch_start
            
            self.history['train_loss'].append(train_loss)
            self.history['train_time'].append(train_time)
            
            # Валидация
            val_start = time.time()
            val_loss, val_f1, val_precision, val_recall, _, _ = self.evaluate_parallel(
                val_loader, id2tag
            )
            val_time = time.time() - val_start
            
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            self.history['val_precision'].append(val_precision)
            self.history['val_recall'].append(val_recall)
            self.history['val_time'].append(val_time)
            
            # Вывод результатов
            print(f"\nРезультаты эпохи {epoch+1}:")
            print(f"  Время обучения:    {train_time:.1f}s")
            print(f"  Время валидации:   {val_time:.1f}s")
            print(f"  Loss обучения:     {train_loss:.4f}")
            print(f"  Loss валидации:    {val_loss:.4f}")
            print(f"  F1-score:          {val_f1:.4f}")
            print(f"  Precision:         {val_precision:.4f}")
            print(f"  Recall:            {val_recall:.4f}")
            
            # Сохранение лучшей модели
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'f1': val_f1,
                }, best_model_path)
                print(f"  ✓ Сохранена лучшая модель (F1: {val_f1:.4f})")
        
        # Загрузка лучшей модели
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\nЗагружена лучшая модель из эпохи {checkpoint['epoch'] + 1}")
        
        return self.history

# Менеджер кеша
class CacheManager:
    """Управление кешированием вычислений"""
    
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    @lru_cache(maxsize=1000)
    def get_cached_result(self, key):
        """Получение результата из кеша"""
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_path):
            try:
                import pickle
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
    
    def cache_result(self, key, result):
        """Сохранение результата в кеш"""
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        import pickle
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)

# Конфигурация с многопоточностью
class ParallelConfig:
    MODEL_NAME = "bert-base-uncased"
    MAX_LENGTH = 128
    BATCH_SIZE = 32  # Увеличенный размер батча
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-5
    DROPOUT_PROB = 0.1
    SEED = 42
    WARMUP_STEPS = 100
    MAX_GRAD_NORM = 1.0
    WEIGHT_DECAY = 0.01
    OUTPUT_DIR = "parallel_results"
    MODEL_SAVE_PATH = "parallel_model.pth"
    
    # Параметры многопоточности
    NUM_WORKERS = 4  # Количество потоков для DataLoader
    PREFETCH_FACTOR = 2  # Предзагрузка данных
    PIN_MEMORY = True  # Фиксация памяти для GPU
    
    # Смешанная точность
    USE_AMP = True  # Использовать автоматическое смешение точности
    
    # Кеширование
    USE_CACHE = True
    CACHE_DIR = "./data_cache"
    
    # Отладка
    DEBUG_MODE = True
    NUM_DEBUG_SAMPLES = 5000

def set_seed(seed):
    """Установка seed для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

def setup_multiprocessing():
    """Настройка многопоточности PyTorch"""
    # Устанавливаем количество потоков для разных операций
    torch.set_num_threads(4)  # Для операций на CPU
    
    # Настройка DataLoader
    num_workers = min(4, os.cpu_count() - 1) if os.cpu_count() > 1 else 0
    
    print(f"Используется {num_workers} воркеров для DataLoader")
    print(f"Доступно CPU ядер: {os.cpu_count()}")
    
    if torch.cuda.is_available():
        print(f"Доступно GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    return num_workers

def main():
    """Основная функция"""
    # Парсинг аргументов
    parser = argparse.ArgumentParser(description='Многопоточная система NER')
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--use_cache', action='store_true', default=True)
    parser.add_argument('--lr', type=float, default=2e-5)
    args = parser.parse_args()
    
    # Конфигурация
    config = ParallelConfig()
    config.DEBUG_MODE = args.debug
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.NUM_WORKERS = args.num_workers
    config.USE_AMP = args.use_amp and torch.cuda.is_available()
    config.USE_CACHE = args.use_cache
    config.LEARNING_RATE = args.lr
    
    # Настройка
    set_seed(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Настройка многопоточности
    num_workers = setup_multiprocessing()
    config.NUM_WORKERS = num_workers
    
    # Создание директории для результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.OUTPUT_DIR, f"experiment_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nУстройство: {device}")
    print(f"Директория результатов: {output_dir}")
    print(f"Размер батча: {config.BATCH_SIZE}")
    print(f"Количество воркеров: {config.NUM_WORKERS}")
    print(f"Смешанная точность: {'Включена' if config.USE_AMP else 'Выключена'}")
    print(f"Кеширование: {'Включено' if config.USE_CACHE else 'Выключено'}")
    
    # Загрузка датасета
    print("\n" + "="*60)
    print("ЗАГРУЗКА ДАННЫХ")
    print("="*60)
    
    start_time = time.time()
    
    try:
        dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")
    except Exception as e:
        print(f"Ошибка загрузки датасета: {e}")
        print("Пробуем загрузить без указания сплита...")
        dataset = load_dataset("DFKI-SLT/few-nerd")
    
    # Информация о тегах
    tag_info = dataset['train'].features['fine_ner_tags'].feature.names
    tag2id = {tag: i for i, tag in enumerate(tag_info)}
    id2tag = {i: tag for i, tag in enumerate(tag_info)}
    
    print(f"Количество тегов: {len(tag_info)}")
    print(f"Обучающая выборка: {len(dataset['train']):,} примеров")
    print(f"Валидационная выборка: {len(dataset['validation']):,} примеров")
    print(f"Тестовая выборка: {len(dataset['test']):,} примеров")
    
    # Сохранение информации о тегах
    with open(os.path.join(output_dir, 'tag_info.json'), 'w') as f:
        json.dump({'tag2id': tag2id, 'id2tag': id2tag}, f, indent=2)
    
    # Токенизатор
    print("\nЗагрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Создание датасетов с кешированием
    print("\nСоздание датасетов...")
    
    if config.USE_CACHE:
        train_dataset = CachedNERDataset(
            dataset['train'], tokenizer, config.MAX_LENGTH,
            cache_dir=config.CACHE_DIR, num_workers=config.NUM_WORKERS,
            debug=config.DEBUG_MODE
        )
        val_dataset = CachedNERDataset(
            dataset['validation'], tokenizer, config.MAX_LENGTH,
            cache_dir=config.CACHE_DIR, num_workers=config.NUM_WORKERS,
            debug=config.DEBUG_MODE
        )
        test_dataset = CachedNERDataset(
            dataset['test'], tokenizer, config.MAX_LENGTH,
            cache_dir=config.CACHE_DIR, num_workers=config.NUM_WORKERS,
            debug=config.DEBUG_MODE
        )
    else:
        # Простые датасеты без кеширования
        class SimpleDataset(Dataset):
            def __init__(self, dataset_split, tokenizer, max_length=128, debug=False):
                self.dataset = dataset_split
                if debug:
                    self.dataset = self.dataset.select(range(min(2000, len(self.dataset))))
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                item = self.dataset[idx]
                tokens = item['tokens']
                labels = item['fine_ner_tags']
                
                encoding = self.tokenizer(
                    tokens,
                    is_split_into_words=True,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                word_ids = encoding.word_ids()
                previous_word_idx = None
                label_ids = []
                
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(labels[word_idx] if word_idx < len(labels) else -100)
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label_ids, dtype=torch.long)
                }
        
        train_dataset = SimpleDataset(dataset['train'], tokenizer, config.MAX_LENGTH, config.DEBUG_MODE)
        val_dataset = SimpleDataset(dataset['validation'], tokenizer, config.MAX_LENGTH, config.DEBUG_MODE)
        test_dataset = SimpleDataset(dataset['test'], tokenizer, config.MAX_LENGTH, config.DEBUG_MODE)
    
    # DataLoader с многопоточностью
    print(f"\nСоздание DataLoader с {config.NUM_WORKERS} воркерами...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
        persistent_workers=config.NUM_WORKERS > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
        persistent_workers=config.NUM_WORKERS > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
        persistent_workers=config.NUM_WORKERS > 0
    )
    
    print(f"Обучающих батчей: {len(train_loader)}")
    print(f"Валидационных батчей: {len(val_loader)}")
    print(f"Тестовых батчей: {len(test_loader)}")
    
    data_load_time = time.time() - start_time
    print(f"Время загрузки данных: {data_load_time:.1f} секунд")
    
    # Создание модели
    print("\n" + "="*60)
    print("СОЗДАНИЕ МОДЕЛИ")
    print("="*60)
    
    model = FastBertCRFForNER(
        config.MODEL_NAME,
        len(tag_info),
        dropout_prob=config.DROPOUT_PROB
    )
    model.to(device)
    
    # Информация о модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    print(f"Размер модели: {total_params * 4 / 1024**2:.1f} MB (FP32)")
    
    if config.USE_AMP and device.type == 'cuda':
        print("Модель будет использовать смешанную точность (FP16/FP32)")
    
    # Обучение
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ МОДЕЛИ")
    print("="*60)
    
    trainer = ParallelNERTrainer(model, config, device)
    
    # Если есть несколько GPU, используем DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Используется {torch.cuda.device_count()} GPU")
        model = nn.DataParallel(model)
    
    training_start = time.time()
    history = trainer.train(train_loader, val_loader, id2tag, output_dir)
    training_time = time.time() - training_start
    
    print(f"\nОбщее время обучения: {training_time:.1f} секунд")
    print(f"Среднее время на эпоху: {training_time / config.NUM_EPOCHS:.1f} секунд")
    
    # Визуализация
    print("\n" + "="*60)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Обучение', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Валидация', marker='s')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Функция потерь')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1-score
    axes[0, 1].plot(history['val_f1'], label='F1-score', color='green', marker='o')
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('F1-score')
    axes[0, 1].set_title('F1-score на валидации')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision/Recall
    axes[0, 2].plot(history['val_precision'], label='Precision', color='orange', marker='o')
    axes[0, 2].plot(history['val_recall'], label='Recall', color='red', marker='s')
    axes[0, 2].set_xlabel('Эпоха')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_title('Precision и Recall')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Время обучения
    axes[1, 0].plot(history['train_time'], label='Обучение', color='blue', marker='o')
    axes[1, 0].plot(history['val_time'], label='Валидация', color='purple', marker='s')
    axes[1, 0].set_xlabel('Эпоха')
    axes[1, 0].set_ylabel('Время (сек)')
    axes[1, 0].set_title('Время выполнения')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Примеры в секунду
    if 'batch_times' in trainer.timing_stats and trainer.timing_stats['batch_times']:
        batch_times = trainer.timing_stats['batch_times']
        samples_per_sec = [config.BATCH_SIZE / t for t in batch_times]
        
        axes[1, 1].plot(samples_per_sec, color='teal', alpha=0.7)
        axes[1, 1].set_xlabel('Номер батча')
        axes[1, 1].set_ylabel('Примеров/сек')
        axes[1, 1].set_title(f'Производительность\nСреднее: {np.mean(samples_per_sec):.1f} примеров/сек')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Использование памяти (если CUDA)
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
        memory_reserved = torch.cuda.max_memory_reserved() / 1024**3
        
        labels = ['Выделено', 'Зарезервировано']
        values = [memory_allocated, memory_reserved]
        
        axes[1, 2].bar(labels, values, color=['skyblue', 'lightcoral'])
        axes[1, 2].set_ylabel('Память (GB)')
        axes[1, 2].set_title(f'Использование GPU памяти\nВсего: {memory_allocated:.2f} GB')
        
        for i, v in enumerate(values):
            axes[1, 2].text(i, v + 0.05, f'{v:.2f} GB', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Тестирование
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("="*60)
    
    test_start = time.time()
    test_loss, test_f1, test_precision, test_recall, test_preds, test_true = trainer.evaluate_parallel(
        test_loader, id2tag
    )
    test_time = time.time() - test_start
    
    print(f"\nРезультаты на тестовом наборе:")
    print(f"  Время тестирования: {test_time:.1f} секунд")
    print(f"  Loss:              {test_loss:.4f}")
    print(f"  F1-score:          {test_f1:.4f}")
    print(f"  Precision:         {test_precision:.4f}")
    print(f"  Recall:            {test_recall:.4f}")
    
    # Детальный отчет
    report = classification_report(test_true, test_preds, digits=4)
    print(f"\nДетальный отчет:\n{report}")
    
    # Сохранение результатов
    print("\n" + "="*60)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    # Сохранение модели и токенизатора
    model_save_path = os.path.join(output_dir, 'model')
    os.makedirs(model_save_path, exist_ok=True)
    
    # Сохраняем состояние модели
    torch.save({
        'model_state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) 
                         else model.module.state_dict(),
        'config': {
            'model_name': config.MODEL_NAME,
            'num_tags': len(tag_info),
            'dropout_prob': config.DROPOUT_PROB
        },
        'tag2id': tag2id,
        'id2tag': id2tag,
        'results': {
            'test_f1': float(test_f1),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_loss': float(test_loss)
        }
    }, os.path.join(model_save_path, 'final_model.pt'))
    
    # Сохраняем токенизатор
    tokenizer.save_pretrained(model_save_path)
    
    # Итоговый отчет
    final_report = {
        'config': {
            'model': config.MODEL_NAME,
            'epochs': config.NUM_EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'num_workers': config.NUM_WORKERS,
            'use_amp': config.USE_AMP,
            'use_cache': config.USE_CACHE
        },
        'results': {
            'test_f1': float(test_f1),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_loss': float(test_loss),
            'total_training_time': float(training_time),
            'data_loading_time': float(data_load_time),
            'test_time': float(test_time)
        },
        'hardware': {
            'device': str(device),
            'cpu_cores': os.cpu_count(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] 
                        if torch.cuda.is_available() else []
        },
        'timing_stats': dict(trainer.timing_stats),
        'history': dict(history)
    }
    
    with open(os.path.join(output_dir, 'final_report.json'), 'w') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    
    # Сохраняем отчет в текстовом формате
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write("МНОГОПОТОЧНАЯ СИСТЕМА NER - РЕЗУЛЬТАТЫ\n")
        f.write("="*60 + "\n\n")
        f.write("КОНФИГУРАЦИЯ:\n")
        f.write(f"  Модель:           {config.MODEL_NAME}\n")
        f.write(f"  Эпохи:            {config.NUM_EPOCHS}\n")
        f.write(f"  Размер батча:     {config.BATCH_SIZE}\n")
        f.write(f"  Learning rate:    {config.LEARNING_RATE}\n")
        f.write(f"  Воркеров:         {config.NUM_WORKERS}\n")
        f.write(f"  Смешанная точность: {'Да' if config.USE_AMP else 'Нет'}\n")
        f.write(f"  Кеширование:      {'Да' if config.USE_CACHE else 'Нет'}\n\n")
        
        f.write("РЕЗУЛЬТАТЫ:\n")
        f.write(f"  F1-score:         {test_f1:.4f}\n")
        f.write(f"  Precision:        {test_precision:.4f}\n")
        f.write(f"  Recall:           {test_recall:.4f}\n")
        f.write(f"  Loss:             {test_loss:.4f}\n\n")
        
        f.write("ПРОИЗВОДИТЕЛЬНОСТЬ:\n")
        f.write(f"  Загрузка данных:  {data_load_time:.1f} сек\n")
        f.write(f"  Обучение:         {training_time:.1f} сек\n")
        f.write(f"  Тестирование:     {test_time:.1f} сек\n")
        f.write(f"  Всего время:      {data_load_time + training_time + test_time:.1f} сек\n\n")
        
        f.write("ДЕТАЛЬНЫЙ ОТЧЕТ:\n")
        f.write(report)
    
    print(f"\nВсе результаты сохранены в: {output_dir}")
    print(f"Модель сохранена в: {model_save_path}")
    
    # Вывод сводки
    print("\n" + "="*60)
    print("СВОДКА ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*60)
    
    total_examples = len(train_dataset) + len(val_dataset) + len(test_dataset)
    total_time = data_load_time + training_time + test_time
    
    print(f"Всего обработано примеров: {total_examples:,}")
    print(f"Общее время выполнения: {total_time:.1f} секунд")
    print(f"Скорость обработки: {total_examples / total_time:.1f} примеров/сек")
    
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Максимальная память GPU: {memory_used:.2f} GB")
    
    print("\n✅ Обучение завершено успешно!")

if __name__ == "__main__":
    # Настройка многопоточности для PyTorch
    mp.set_start_method('spawn', force=True)  # Для совместимости с CUDA
    
    # Запуск
    main()