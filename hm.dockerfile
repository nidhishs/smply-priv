FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

COPY human-masking/requirements.txt /tmp/requirements.txt

# Install git to clone detectron2 and build.
RUN apt-get update && apt-get install -y git && \
    pip install -r /tmp/requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY human-masking /app

WORKDIR /app

ENTRYPOINT [ "python", "run_hm.py" ]