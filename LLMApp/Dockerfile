FROM python:3.11-slim
WORKDIR /app

# Kopiraj datoteke iz korena (ker je context: .), Dockerfile je v LLMApp/
COPY disk_model.pkl .
COPY feature_importance.csv .
COPY LLMApp/requirements.txt .

RUN pip install -r requirements.txt
# Kopiraj vsebino mape LLMApp v kontejner
COPY LLMApp/ .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]