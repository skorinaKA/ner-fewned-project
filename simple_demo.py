"""
Упрощенная версия NER для быстрого запуска
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from seqeval.metrics import classification_report, f1_score
import random
from tqdm import tqdm
import os

# Упрощенная модель без CRF
class BertForNER(nn.Module):
    def __init__(self, model_name, num_tags):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.shape[-1])[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss, logits
        return logits

# Dataset
class SimpleNERDataset(Dataset):
    def __init__(self, dataset_split, tokenizer, max_length=128, num_samples=1000):
        self.dataset = dataset_split.select(range(min(num_samples, len(dataset_split))))
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        tokens = item['tokens'][:30]  # Берем первые 30 токенов
        labels = item['fine_ner_tags'][:30]
        
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

def main():
    print("Запуск упрощенной версии NER...")
    
    # Настройки
    MODEL_NAME = "bert-base-uncased"
    BATCH_SIZE = 8
    EPOCHS = 2
    LR = 2e-5
    MAX_LENGTH = 128
    NUM_SAMPLES = 1000  # Для быстрого обучения
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    # Загрузка данных
    print("Загрузка датасета...")
    dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")
    
    # Информация о тегах
    tag_info = dataset['train'].features['fine_ner_tags'].feature.names
    id2tag = {i: tag for i, tag in enumerate(tag_info)}
    print(f"Количество тегов: {len(tag_info)}")
    
    # Токенизатор
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Датасеты
    train_dataset = SimpleNERDataset(dataset['train'], tokenizer, MAX_LENGTH, NUM_SAMPLES)
    val_dataset = SimpleNERDataset(dataset['validation'], tokenizer, MAX_LENGTH, NUM_SAMPLES // 5)
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Модель
    model = BertForNER(MODEL_NAME, len(tag_info))
    model.to(device)
    
    # Оптимизатор
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # Обучение
    print("\nОбучение модели...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Эпоха {epoch+1}/{EPOCHS}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Loss эпохи {epoch+1}: {total_loss/len(train_loader):.4f}")
        
        # Валидация
        model.eval()
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=-1)
                
                # Преобразование в теги
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
                    
                    all_preds.append(pred_tags)
                    all_true.append(true_tags)
        
        f1 = f1_score(all_true, all_preds)
        print(f"F1 на валидации: {f1:.4f}")
    
    # Сохранение модели
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), 'results/simple_model.pth')
    print("\nМодель сохранена в results/simple_model.pth")
    
    # Пример предсказания
    print("\nПример предсказания:")
    model.eval()
    test_sample = dataset['test'][0]
    tokens = test_sample['tokens'][:2000]
    
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        logits = model(encoding['input_ids'].to(device), encoding['attention_mask'].to(device))
        predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    
    word_ids = encoding.word_ids()
    previous_word_idx = None
    
    print("Текст:", ' '.join(tokens))
    print("Предсказанные сущности:")
    
    for i, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            tag = id2tag[predictions[i]]
            if tag != 'O':
                print(f"  {tokens[word_idx]}: {tag}")
        previous_word_idx = word_idx

if __name__ == "__main__":
    main()