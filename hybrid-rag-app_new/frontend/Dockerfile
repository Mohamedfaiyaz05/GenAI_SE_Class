FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/frontend

COPY frontend/requirements.txt /app/frontend/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/frontend/requirements.txt

COPY frontend/ /app/frontend/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
