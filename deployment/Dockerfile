# SuperKart sales forecast - Streamlit app, deployed as a Hugging Face Space (SDK: docker)

FROM python:3.10-slim

WORKDIR /app

# System deps for scientific python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Hugging Face Spaces expect the app to listen on port 7860
EXPOSE 7860

ENV MODEL_REPO_ID="your-hf-username/superkart-sales-model"

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
