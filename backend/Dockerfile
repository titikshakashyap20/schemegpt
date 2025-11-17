FROM python:3.11-slim
WORKDIR /app
COPY backend_app /app/backend_app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
CMD ["uvicorn", "backend_app.main:app", "--host", "0.0.0.0", "--port", "8001"]
