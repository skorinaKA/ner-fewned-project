import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from torchcrf import CRF
from datasets import load_dataset
import numpy as np
import pandas as pd
from collections import defaultdict
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Установка seed для воспроизводимости
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Проверка доступности GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Загрузка датасета
print("Loading Few-NERD dataset...")
dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")

# Посмотрим на структуру датасета
print(f"Dataset structure: {dataset}")
print(f"Train samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['validation'])}")
print(f"Test samples: {len(dataset['test'])}")

# Посмотрим на пример из датасета
sample = dataset['train'][0]
print("\nSample from dataset:")
print(f"Tokens: {sample['tokens'][:20]}...")  # первые 20 токенов
print(f"NER tags: {sample['fine_ner_tags'][:20]}...")
print(f"Number of tokens in sample: {len(sample['tokens'])}")

# Создадим словарь для соответствия тегов и их названий
# В Few-NERD используется 8 грубых типов и 66 вложенных типов
tag_info = dataset['train'].features['fine_ner_tags'].feature.names
print(f"\nTotal number of NER tags: {len(tag_info)}")

# Для удобства создадим маппинг тегов
tag2id = {tag: i for i, tag in enumerate(tag_info)}
id2tag = {i: tag for i, tag in enumerate(tag_info)}

print(f"\nFirst 10 tags: {tag_info[:10]}")

class NERDataset(Dataset):
    def __init__(self, dataset_split, tokenizer, max_length=128):
        self.dataset = dataset_split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tag2id = tag2id
        self.id2tag = id2tag
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        tokens = item['tokens']
        labels = item['fine_ner_tags']
        
        # Токенизация с выравниванием меток
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Выравнивание меток для подтокенов
        word_ids = encoding.word_ids()
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                # Специальные токены ([CLS], [SEP], [PAD])
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Первый токен слова
                label_ids.append(labels[word_idx])
            else:
                # Последующие токены того же слова
                # Используем -100 для игнорирования при вычислении потерь
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        # Преобразуем в тензоры
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }
        
        return item

# Инициализация токенизатора
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Создание датасетов
train_dataset = NERDataset(dataset['train'], tokenizer)
val_dataset = NERDataset(dataset['validation'], tokenizer)
test_dataset = NERDataset(dataset['test'], tokenizer)

# Создание DataLoader
BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

class BertCRFForNER(nn.Module):
    def __init__(self, model_name, num_tags, dropout_prob=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Получаем эмбеддинги от BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Применяем dropout и линейный слой
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        
        # Маска для игнорирования паддинга
        mask = attention_mask.bool()
        
        if labels is not None:
            # Рассчитываем loss через CRF
            log_likelihood = self.crf(emissions, labels, mask=mask, reduction='mean')
            loss = -log_likelihood
            return loss, emissions
        else:
            # Предсказание с помощью Viterbi декодера
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions

# Инициализация модели
num_tags = len(tag_info)
model = BertCRFForNER(MODEL_NAME, num_tags)
model.to(device)

print(f"Model architecture:")
print(model)
print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, lr=2e-5, num_epochs=3):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # Оптимизатор
        bert_params = list(model.bert.named_parameters())
        classifier_params = list(model.classifier.named_parameters()) + list(model.crf.named_parameters())
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
                'lr': lr,
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)],
                'lr': lr,
                'weight_decay': 0.0
            },
            {
                'params': [p for n, p in classifier_params if not any(nd in n for nd in no_decay)],
                'lr': lr * 10,
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in classifier_params if any(nd in n for nd in no_decay)],
                'lr': lr * 10,
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(optimizer_grouped_parameters)
        
        # Планировщик
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * 0.1)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # История обучения
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': []
        }
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch in pbar:
            # Перемещаем данные на устройство
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Обнуляем градиенты
            self.optimizer.zero_grad()
            
            # Forward pass
            loss, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Backward pass
            loss.backward()
            
            # Обрезание градиентов
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Шаг оптимизации
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Вычисляем loss
                loss, emissions = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += loss.item()
                
                # Получаем предсказания
                predictions = model.crf.decode(emissions, mask=attention_mask.bool())
                
                # Преобразуем в списки тегов
                for i in range(len(predictions)):
                    pred_seq = predictions[i]
                    true_seq = labels[i].cpu().numpy()
                    mask = attention_mask[i].cpu().numpy()
                    
                    # Фильтруем паддинг и специальные токены
                    pred_tags = []
                    true_tags = []
                    
                    for j in range(len(pred_seq)):
                        if mask[j] == 1 and true_seq[j] != -100:
                            pred_tags.append(id2tag[pred_seq[j]])
                            true_tags.append(id2tag[true_seq[j]])
                    
                    all_predictions.append(pred_tags)
                    all_true_labels.append(true_tags)
        
        # Вычисляем метрики
        f1 = f1_score(all_true_labels, all_predictions)
        precision = precision_score(all_true_labels, all_predictions)
        recall = recall_score(all_true_labels, all_predictions)
        
        avg_loss = total_loss / len(loader)
        
        return avg_loss, f1, precision, recall, all_predictions, all_true_labels
    
    def train(self):
        best_f1 = 0
        best_model_state = None
        
        for epoch in range(self.num_epochs):
            # Обучение
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            
            # Валидация
            val_loss, val_f1, val_precision, val_recall, _, _ = self.evaluate(self.val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            
            print(f"\nEpoch {epoch+1}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val F1: {val_f1:.4f}")
            print(f"  Val Precision: {val_precision:.4f}")
            print(f"  Val Recall: {val_recall:.4f}")
            
            # Сохраняем лучшую модель
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, 'best_model.pth')
                print(f"  ✓ Saved new best model with F1: {val_f1:.4f}")
        
        # Загружаем лучшую модель
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return self.history

# Обучение модели
print("Starting training...")
trainer = Trainer(model, train_loader, val_loader, device, lr=2e-5, num_epochs=3)
history = trainer.train()

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # График loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # График F1-score
    axes[1].plot(history['val_f1'], label='Val F1-score', marker='o', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1-score')
    axes[1].set_title('Validation F1-score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# Визуализация
plot_training_history(history)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Получаем предсказания
            predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Преобразуем в списки тегов
            for i in range(len(predictions)):
                pred_seq = predictions[i]
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
    
    # Подробный отчет по метрикам
    report = classification_report(all_true_labels, all_predictions, digits=4)
    
    # Группируем по основным категориям для анализа
    def get_main_category(tag):
        if tag == 'O':
            return 'O'
        else:
            # В Few-NERD теги вида "B-art-broadcastprogram"
            parts = tag.split('-')
            if len(parts) > 1:
                return parts[1]  # Возвращаем основную категорию (art, building, etc.)
            return tag
    
    # Создаем confusion matrix для основных категорий
    main_true = [[get_main_category(tag) for tag in seq] for seq in all_true_labels]
    main_pred = [[get_main_category(tag) for tag in seq] for seq in all_predictions]
    
    # Уникальные категории
    all_categories = sorted(set([tag for seq in main_true for tag in seq] + 
                                [tag for seq in main_pred for tag in seq]))
    
    # Вычисляем метрики
    f1 = f1_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions)
    recall = recall_score(all_true_labels, all_predictions)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'report': report,
        'predictions': all_predictions,
        'true_labels': all_true_labels,
        'main_true': main_true,
        'main_pred': main_pred,
        'categories': all_categories
    }

# Оценка модели
print("\nEvaluating on test set...")
results = evaluate_model(model, test_loader, device)

print("\n" + "="*50)
print("TEST RESULTS")
print("="*50)
print(f"F1-score: {results['f1']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")
print("\nDetailed classification report:")
print(results['report'])

def visualize_results(results, top_n=15):
    # 1. Confusion matrix для основных категорий
    # Собираем все теги
    all_true_flat = [tag for seq in results['main_true'] for tag in seq]
    all_pred_flat = [tag for seq in results['main_pred'] for tag in seq]
    
    # Создаем confusion matrix
    cm = confusion_matrix(all_true_flat, all_pred_flat, labels=results['categories'])
    
    # Нормализуем по строкам (истинные метки)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Ограничиваем количество категорий для визуализации
    if len(results['categories']) > top_n:
        # Выбираем top_n наиболее частых категорий
        category_counts = {}
        for tag in all_true_flat:
            category_counts[tag] = category_counts.get(tag, 0) + 1
        
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_categories = [cat for cat, _ in top_categories]
        
        # Фильтруем confusion matrix
        indices = [results['categories'].index(cat) for cat in top_categories]
        cm_filtered = cm[np.ix_(indices, indices)]
        cm_normalized_filtered = cm_normalized[np.ix_(indices, indices)]
        
        categories_to_show = top_categories
        cm_to_show = cm_normalized_filtered
    else:
        categories_to_show = results['categories']
        cm_to_show = cm_normalized
    
    # Визуализация confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Нормализованная confusion matrix
    im = axes[0].imshow(cm_to_show, cmap='Blues', aspect='auto')
    axes[0].set_xticks(np.arange(len(categories_to_show)))
    axes[0].set_yticks(np.arange(len(categories_to_show)))
    axes[0].set_xticklabels(categories_to_show, rotation=45, ha='right')
    axes[0].set_yticklabels(categories_to_show)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Normalized Confusion Matrix')
    
    # Добавляем значения в ячейки
    for i in range(len(categories_to_show)):
        for j in range(len(categories_to_show)):
            text = axes[0].text(j, i, f'{cm_to_show[i, j]:.2f}',
                           ha="center", va="center", 
                           color="white" if cm_to_show[i, j] > 0.5 else "black")
    
    plt.colorbar(im, ax=axes[0])
    
    # 2. F1-score по категориям (из отчета)
    # Парсим отчет для получения F1 по категориям
    lines = results['report'].split('\n')
    category_metrics = {}
    
    for line in lines[2:-5]:  # Пропускаем заголовок и summary
        if line.strip():
            parts = line.split()
            if len(parts) >= 5:
                category = parts[0]
                f1 = float(parts[3])
                category_metrics[category] = f1
    
    # Сортируем категории по F1-score
    sorted_categories = sorted(category_metrics.items(), key=lambda x: x[1], reverse=True)
    categories_sorted = [cat for cat, _ in sorted_categories[:15]]
    f1_scores_sorted = [score for _, score in sorted_categories[:15]]
    
    # Bar plot для F1-score
    bars = axes[1].barh(range(len(categories_sorted)), f1_scores_sorted, color='skyblue')
    axes[1].set_yticks(range(len(categories_sorted)))
    axes[1].set_yticklabels(categories_sorted)
    axes[1].set_xlabel('F1-score')
    axes[1].set_title('Top 15 Categories by F1-score')
    axes[1].invert_yaxis()  # Наибольший F1 сверху
    
    # Добавляем значения на bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[1].text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('results_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm_to_show, categories_to_show

# Визуализация
cm, categories = visualize_results(results)

def show_examples(model, dataset, tokenizer, num_examples=5):
    model.eval()
    indices = random.sample(range(len(dataset)), num_examples)
    
    for idx in indices:
        sample = dataset.dataset[idx]
        tokens = sample['tokens'][:50]  # Берем первые 50 токенов для наглядности
        true_labels = sample['fine_ner_tags'][:50]
        
        # Токенизация
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        # Предсказание
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            predictions = model(input_ids=input_ids, attention_mask=attention_mask)[0]
            
            # Выравниваем предсказания
            word_ids = encoding.word_ids()
            previous_word_idx = None
            pred_labels = []
            
            for word_idx in word_ids:
                if word_idx is None or word_idx >= len(tokens):
                    break
                elif word_idx != previous_word_idx:
                    pred_labels.append(id2tag[predictions[word_idx]])
                previous_word_idx = word_idx
        
        # Выводим результат
        print(f"\nExample {idx + 1}:")
        print("-" * 80)
        
        current_entity = None
        current_text = []
        
        for i, (token, true_tag, pred_tag) in enumerate(zip(tokens, true_labels[:len(pred_labels)], pred_labels)):
            true_tag_name = id2tag[true_tag]
            
            if true_tag_name.startswith('B-'):
                if current_entity:
                    print(f"  True: [{' '.join(current_text)}] as {current_entity}")
                    current_text = []
                current_entity = true_tag_name[2:]
                current_text.append(token)
            elif true_tag_name.startswith('I-'):
                current_text.append(token)
            else:
                if current_entity:
                    print(f"  True: [{' '.join(current_text)}] as {current_entity}")
                    current_text = []
                    current_entity = None
        
        if current_entity:
            print(f"  True: [{' '.join(current_text)}] as {current_entity}")
        
        # Аналогично для предсказаний
        current_entity = None
        current_text = []
        
        for i, (token, pred_tag) in enumerate(zip(tokens, pred_labels)):
            if pred_tag.startswith('B-'):
                if current_entity:
                    print(f"  Pred: [{' '.join(current_text)}] as {current_entity}")
                    current_text = []
                current_entity = pred_tag[2:]
                current_text.append(token)
            elif pred_tag.startswith('I-'):
                current_text.append(token)
            else:
                if current_entity:
                    print(f"  Pred: [{' '.join(current_text)}] as {current_entity}")
                    current_text = []
                    current_entity = None
        
        if current_entity:
            print(f"  Pred: [{' '.join(current_text)}] as {current_entity}")
        
        print("-" * 80)

# Показываем примеры
print("\n" + "="*80)
print("EXAMPLES OF PREDICTIONS")
print("="*80)
show_examples(model, test_dataset, tokenizer, num_examples=3)

def save_model(model, tokenizer, tag2id, id2tag, path='ner_model'):
    # Сохраняем модель
    torch.save({
        'model_state_dict': model.state_dict(),
        'tag2id': tag2id,
        'id2tag': id2tag,
        'model_name': MODEL_NAME,
        'num_tags': num_tags
    }, f'{path}_full.pth')
    
    # Сохраняем токенизатор
    tokenizer.save_pretrained(path)
    
    print(f"Model saved to {path}_full.pth")
    print(f"Tokenizer saved to {path}/")

def load_model(path='ner_model'):
    # Загружаем конфигурацию
    checkpoint = torch.load(f'{path}_full.pth', map_location=device)
    
    # Создаем модель
    model = BertCRFForNER(checkpoint['model_name'], checkpoint['num_tags'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    return model, tokenizer, checkpoint['tag2id'], checkpoint['id2tag']

# Сохраняем модель
save_model(model, tokenizer, tag2id, id2tag)

# Пример загрузки
# loaded_model, loaded_tokenizer, loaded_tag2id, loaded_id2tag = load_model()

def analyze_errors(true_labels, predictions):
    errors = []
    
    for true_seq, pred_seq in zip(true_labels, predictions):
        for true_tag, pred_tag in zip(true_seq, pred_seq):
            if true_tag != pred_tag and true_tag != 'O':
                errors.append({
                    'true': true_tag,
                    'pred': pred_tag,
                    'type': 'incorrect' if pred_tag != 'O' else 'missed'
                })
            elif true_tag == 'O' and pred_tag != 'O':
                errors.append({
                    'true': true_tag,
                    'pred': pred_tag,
                    'type': 'false_positive'
                })
    
    # Анализ типов ошибок
    error_df = pd.DataFrame(errors)
    
    if len(error_df) > 0:
        print("\nError Analysis:")
        print("="*50)
        
        # Распределение типов ошибок
        print("\nError Types Distribution:")
        print(error_df['type'].value_counts())
        
        # Наиболее частые путаницы
        print("\nMost Common Confusions:")
        confusions = error_df[error_df['type'] == 'incorrect'].groupby(['true', 'pred']).size().reset_index(name='count')
        confusions = confusions.sort_values('count', ascending=False).head(10)
        print(confusions)
        
        # Визуализация
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Типы ошибок
        error_types = error_df['type'].value_counts()
        axes[0].pie(error_types.values, labels=error_types.index, autopct='%1.1f%%')
        axes[0].set_title('Distribution of Error Types')
        
        # Top путаницы
        top_confusions = confusions.head(8)
        axes[1].barh(range(len(top_confusions)), top_confusions['count'])
        axes[1].set_yticks(range(len(top_confusions)))
        axes[1].set_yticklabels([f"{row['true']} → {row['pred']}" for _, row in top_confusions.iterrows()])
        axes[1].set_xlabel('Count')
        axes[1].set_title('Top 8 Most Common Confusions')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return error_df

# Анализ ошибок
error_df = analyze_errors(results['true_labels'], results['predictions'])

def create_final_report(results, history, model, error_df):
    report = """
    FINAL REPORT: NER on Few-NERD Dataset
    ======================================
    
    1. MODEL ARCHITECTURE:
    ----------------------
    - Base Model: BERT-base-uncased
    - Additional Layers: Linear Classifier + CRF
    - Number of parameters: {:,}
    - Number of trainable parameters: {:,}
    
    2. TRAINING DETAILS:
    --------------------
    - Batch size: {}
    - Learning rate: 2e-5 (BERT), 2e-4 (classifier/CRF)
    - Optimizer: AdamW with weight decay
    - Scheduler: Linear warmup with decay
    - Number of epochs: {}
    
    3. RESULTS:
    -----------
    - Final Test F1-score: {:.4f}
    - Final Test Precision: {:.4f}
    - Final Test Recall: {:.4f}
    
    4. ERROR ANALYSIS:
    ------------------
    - Total errors analyzed: {}
    - Most common error type: {}
    - Most frequent confusion: {}
    
    5. KEY OBSERVATIONS:
    --------------------
    1. The model achieves competitive performance on the Few-NERD dataset.
    2. CRF layer helps maintain label consistency in predictions.
    3. Some entity types (like rare or nested ones) remain challenging.
    4. The model sometimes confuses similar entity types.
    
    6. RECOMMENDATIONS FOR IMPROVEMENT:
    ------------------------------------
    1. Use larger pretrained models (BERT-large, RoBERTa).
    2. Implement handling for nested entities.
    3. Add character-level embeddings for better handling of OOV tokens.
    4. Use ensemble methods.
    5. Apply data augmentation techniques.
    
    """.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        BATCH_SIZE,
        len(history['train_loss']),
        results['f1'],
        results['precision'],
        results['recall'],
        len(error_df) if error_df is not None else 0,
        error_df['type'].value_counts().index[0] if error_df is not None and len(error_df) > 0 else 'N/A',
        error_df.groupby(['true', 'pred']).size().idxmax() if error_df is not None and len(error_df) > 0 else 'N/A'
    )
    
    # Сохраняем отчет в файл
    with open('final_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    print("Full report saved to 'final_report.txt'")

# Создаем итоговый отчет
create_final_report(results, history, model, error_df)