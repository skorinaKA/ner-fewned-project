"""
Скрипт для установки всех зависимостей
"""

import subprocess
import sys
import os

def install_packages():
    """Установка всех необходимых пакетов"""
    
    packages = [
        "torch>=1.9.0",
        "transformers>=4.10.0",
        "datasets>=2.0.0",
        "seqeval>=1.2.2",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "threadpoolctl>=3.1.0"
    ]
    
    print("Установка зависимостей для многопоточной системы NER...")
    print("="*60)
    
    for package in packages:
        print(f"\nУстановка {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"✓ {package} установлен успешно")
        except subprocess.CalledProcessError as e:
            print(f"✗ Ошибка установки {package}: {e}")
    
    print("\n" + "="*60)
    print("Все зависимости установлены!")
    
    # Проверка версий
    print("\nПроверка версий установленных пакетов:")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
    except:
        print("  PyTorch: НЕ УСТАНОВЛЕН")
    
    try:
        import transformers
        print(f"  Transformers: {transformers.__version__}")
    except:
        print("  Transformers: НЕ УСТАНОВЛЕН")
    
    try:
        import datasets
        print(f"  Datasets: {datasets.__version__}")
    except:
        print("  Datasets: НЕ УСТАНОВЛЕН")

def check_cuda():
    """Проверка доступности CUDA"""
    print("\n" + "="*60)
    print("ПРОВЕРКА АППАРАТНОГО ОБЕСПЕЧЕНИЯ")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print("✓ CUDA доступен")
            print(f"  CUDA версия: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("✗ CUDA недоступен, будет использоваться CPU")
    except:
        print("  Не удалось проверить CUDA")

def main():
    """Основная функция"""
    print("МНОГОПОТОЧНАЯ СИСТЕМА NER - УСТАНОВКА")
    print("="*60)
    
    # Установка пакетов
    install_packages()
    
    # Проверка CUDA
    check_cuda()
    
    print("\n" + "="*60)
    print("УСТАНОВКА ЗАВЕРШЕНА!")
    print("="*60)
    print("\nТеперь вы можете запустить систему:")
    print("  python fast_ner_fixed.py --debug --epochs 2 --batch_size 16")

if __name__ == "__main__":
    main()