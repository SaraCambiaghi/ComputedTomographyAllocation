FROM python:3.10

RUN apt-get update && apt-get install -y \
    python3-pip \
    wget \
    unzip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

RUN chmod +x ./cbc

EXPOSE 5000

CMD ["python", "app.py"]
