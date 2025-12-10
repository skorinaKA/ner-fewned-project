"""
Основной скрипт для запуска NER системы через Docker
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Запуск NER системы в Docker')
    
    # Основные параметры
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'evaluate', 'predict', 'test'],
                       help='Режим работы: train, evaluate, predict, test')
    
    # Параметры обучения
    parser.add_argument('--epochs', type=int, default=3, help='Количество эпох')
    parser.add_argument('--batch_size', type=int, default=16, help='Размер батча')
    parser.add_argument('--workers', type=int, default=4, help='Количество воркеров')
    parser.add_argument('--model', type=str, default='bert-base-uncased', 
                       help='Название модели')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=5000, 
                       help='Максимальное количество примеров')
    
    # Пути
    parser.add_argument('--data_dir', type=str, default='/app/data', 
                       help='Директория с данными')
    parser.add_argument('--results_dir', type=str, default='/app/results', 
                       help='Директория для результатов')
    parser.add_argument('--model_dir', type=str, default='/app/models', 
                       help='Директория с моделями')
    
    # Дополнительные параметры
    parser.add_argument('--debug', action='store_true', help='Режим отладки')
    parser.add_argument('--no_cache', action='store_true', 
                       help='Не использовать кеширование')
    parser.add_argument('--gpu', action='store_true', help='Использовать GPU')
    
    args = parser.parse_args()
    
    # Проверяем доступность GPU
    if args.gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                print("⚠️  GPU запрошен, но не доступен. Используется CPU.")
                args.gpu = False
        except:
            print("⚠️  Не удалось проверить доступность GPU")
            args.gpu = False
    
    # Создаем директории если их нет
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # В зависимости от режима запускаем соответствующий скрипт
    if args.mode == 'train':
        from ner_system import main as train_main
        train_args = argparse.Namespace(
            debug=args.debug,
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            model=args.model,
            lr=args.lr,
            max_samples=args.max_samples,
            no_amp=not args.gpu,
            max_length=128
        )
        train_main()
    
    elif args.mode == 'evaluate':
        print("Режим оценки модели")
        # Добавьте код оценки
        
    elif args.mode == 'predict':
        print("Режим предсказания")
        # Добавьте код предсказания
        
    elif args.mode == 'test':
        print("Тестовый режим")
        from simple_ner import main as test_main
        test_main()
    
    print(f"\n✅ Работа завершена. Результаты в: {args.results_dir}")

if __name__ == "__main__":
    main()