"""
Многопоточная система NER для Docker
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import random
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import warnings
import argparse
import os
from datetime import datetime
import time

warnings.filterwarnings('ignore')

class SimpleBertForNER(nn.Module):
    def __init__(self, model_name, num_tags, dropout_prob=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
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

class FastNERDataset(Dataset):
    def __init__(self, dataset_split, tokenizer, max_length=128, max_samples=5000):
        self.dataset = dataset_split.select(range(min(max_samples, len(dataset_split))))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._preprocess()
    
    def _preprocess(self):
        results = []
        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            tokens = item['tokens'][:self.max_length-2]
            labels = item['fine_ner_tags'][:self.max_length-2]
            
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
                    if word_idx < len(labels):
                        label_ids.append(labels[word_idx])
                    else:
                        label_ids.append(-100)
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            results.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label_ids, dtype=torch.long)
            })
        
        return results
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def save_training_plots(history, output_dir):
    """Сохранение графиков обучения"""
    if not history:
        return
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train', marker='o')
    plt.plot(history['val_loss'], label='Val', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['val_f1'], label='F1', color='green', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.title('F1-score')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['val_precision'], label='Precision', color='orange', marker='o')
    plt.plot(history['val_recall'], label='Recall', color='red', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Precision & Recall')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--max_samples', type=int, default=5000)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='/app/results')
    
    args = parser.parse_args()
    
    # Настройка
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создание директории
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    
    # Загрузка данных
    print("Loading dataset...")
    try:
        dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")
    except:
        dataset = load_dataset("DFKI-SLT/few-nerd")
    
    # Теги
    tag_info = dataset['train'].features['fine_ner_tags'].feature.names
    id2tag = {i: tag for i, tag in enumerate(tag_info)}
    tag2id = {tag: i for i, tag in enumerate(tag_info)}
    
    # Токенизатор
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Датасеты
    train_dataset = FastNERDataset(dataset['train'], tokenizer, args.max_length, args.max_samples)
    val_dataset = FastNERDataset(dataset['validation'], tokenizer, args.max_length, args.max_samples // 5)
    test_dataset = FastNERDataset(dataset['test'], tokenizer, args.max_length, args.max_samples // 10)
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Модель
    model = SimpleBertForNER(args.model, len(tag_info))
    model.to(device)
    
    # Обучение
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=100, 
        num_training_steps=len(train_loader) * args.epochs
    )
    
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_precision': [], 'val_recall': []}
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        history['train_loss'].append(total_loss / len(train_loader))
        
        # Validation
        model.eval()
        all_preds, all_true = [], []
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                loss, logits = model(input_ids, attention_mask, labels)
                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                
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
        
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_f1'].append(f1_score(all_true, all_preds))
        history['val_precision'].append(precision_score(all_true, all_preds))
        history['val_recall'].append(recall_score(all_true, all_preds))
        
        print(f"Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"Val F1: {history['val_f1'][-1]:.4f}")
    
    # Сохранение
    save_training_plots(history, output_dir)
    
    # Сохраняем модель
    model_dir = os.path.join(output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(model_dir, 'model_weights.pth'))
    tokenizer.save_pretrained(model_dir)
    
    with open(os.path.join(model_dir, 'tags.json'), 'w') as f:
        json.dump({'id2tag': id2tag, 'tag2id': tag2id}, f, indent=2)
    
    # Сохраняем результаты
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({
            'history': history,
            'config': vars(args)
        }, f, indent=2)
    
    print(f"\n✅ Training complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()