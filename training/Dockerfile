FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

COPY . /app
WORKDIR /app

CMD ["python"]