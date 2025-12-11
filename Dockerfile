FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Создаем папку для шаблонов
RUN mkdir -p /app/templates

EXPOSE 5000
EXPOSE 8888

CMD ["python", "app.py"]