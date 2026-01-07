FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app4.py .
COPY *.pkl ./
COPY *.png ./

RUN mkdir -p /app/.streamlit && \
    echo "[server]" > /app/.streamlit/config.toml && \
    echo "port = 8501" >> /app/.streamlit/config.toml && \
    echo "address = \"0.0.0.0\"" >> /app/.streamlit/config.toml && \
    echo "headless = true" >> /app/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app4.py"]