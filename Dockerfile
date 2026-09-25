FROM python:3.12-slim

WORKDIR /opt/project

COPY requirements.txt .

RUN pip install --no-cache-dir --index-url https://pypi.org/simple -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/project

CMD ["tail", "-f", "/dev/null"]