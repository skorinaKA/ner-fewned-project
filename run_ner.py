"""
Скрипт быстрого запуска NER системы
"""

import subprocess
import sys
import os

def install_dependencies():
    """Установка минимальных зависимостей"""
    print("Установка зависимостей...")
    packages = [
        "torch",
        "transformers",
        "datasets",
        "seqeval",
        "numpy",
        "matplotlib",
        "tqdm"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"✓ {package}")
        except:
            print(f"✗ Ошибка установки {package}")

def main():
    print("="*60)
    print("БЫСТРЫЙ ЗАПУСК СИСТЕМЫ NER")
    print("="*60)
    
    # Установка зависимостей
    install_dependencies()
    
    # Создание файла с кодом если его нет
    if not os.path.exists("ner_system_final.py"):
        print("\nСоздайте файл ner_system_final.py с кодом выше")
        return
    
    # Запуск с оптимальными параметрами для быстрого теста
    cmd = [
        sys.executable, "ner_system_final.py",
        "--debug",
        "--epochs", "2",
        "--batch_size", "8",
        "--workers", "2",
        "--max_samples", "1000"
    ]
    
    print(f"\nЗапуск команды: {' '.join(cmd)}")
    print("="*60)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем")
    except Exception as e:
        print(f"\nОшибка: {e}")

if __name__ == "__main__":
    main()