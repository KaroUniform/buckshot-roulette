# Используем официальный образ Python 3.11
FROM python:3.11-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл requirements из локальной директории app в контейнер в директорию /app/requirements
COPY app/requirements.txt /app/requirements.txt

# Устанавливаем зависимости из файла requirements
RUN pip install --no-cache-dir -r /app/requirements.txt

# Копируем все содержимое локальной директории app в контейнер в директорию /app
COPY app /app

# Запускаем проект при старте контейнера
CMD ["python", "main.py"]
