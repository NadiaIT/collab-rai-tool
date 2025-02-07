FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

COPY ./app/backend_baseline_version/requirements.txt .

RUN pip3 install -r requirements.txt

COPY ./app/backend_baseline_version ./backend_baseline_version
COPY ./app/helpers ./helpers

EXPOSE 8505

CMD ["python", "backend_baseline_version/main.py"]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8502" ,"--reload"]