FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860 MODEL_REPO_ID=dinzprof/superkart-sales-model REFERENCE_YEAR=2026
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 7860
CMD ["streamlit","run","app.py","--server.address=0.0.0.0","--server.port=7860","--server.headless=true"]
