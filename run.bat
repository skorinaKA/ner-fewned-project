@echo off

REM Установка зависимостей
pip install -r requirements.txt

REM Создание директории для результатов
if not exist results mkdir results

REM Запуск с параметрами по умолчанию
echo Запуск с параметрами по умолчанию...
python main.py

REM Запуск в режиме отладки
REM echo Запуск в режиме отладки...
REM python main.py --debug --epochs 1 --batch_size 8

REM Запуск с другими параметрами
REM echo Запуск с увеличенным количеством эпох...
REM python main.py --epochs 5 --batch_size 32 --lr 3e-5

pause