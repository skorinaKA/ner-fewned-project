@echo off

echo Установка зависимостей...
pip install torch transformers datasets seqeval numpy tqdm matplotlib

echo Запуск минимальной версии...
python simple_ner_minimal.py

echo Запуск полной версии...
python ner_system_fixed.py --debug --epochs 2 --batch_size 16 --workers 4

pause