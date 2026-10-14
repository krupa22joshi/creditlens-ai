FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Train at build time so containers start instantly with a versioned model.
RUN python scripts/train.py

EXPOSE 8000 8501
CMD ["uvicorn", "creditlens.api:app", "--host", "0.0.0.0", "--port", "8000"]
