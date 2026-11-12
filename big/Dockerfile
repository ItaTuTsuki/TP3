# Image de l'app Streamlit (partie 2) — légère, sans Spark : lit app/data/*.csv.gz
#   docker build -t tp3-transfermarkt-app .
#   docker run -p 8501:8501 tp3-transfermarkt-app   -> http://localhost:8501
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
