FROM python:3.14.0 as buider

WORKDIR /app

#install

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq \
    && rm  -rf /var/lib/apt/lists/*

#copy  on  dependency \
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
