FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip --root-user-action=ignore
RUN pip install --no-cache-dir -r requirements.txt --root-user-action=ignore

COPY . .

EXPOSE 8501

CMD ["sleep", "infinity"]
