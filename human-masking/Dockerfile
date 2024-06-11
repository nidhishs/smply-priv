FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

COPY requirements.txt /tmp/requirements.txt

# Install mmcv before opencv-python-headless to avoid conflict with opencv-python.
RUN pip install --no-cache-dir \
    mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html \
    mmengine==0.10.4 && \
    pip uninstall -y opencv-python

# Install git to build detectron2 from source.
RUN apt-get update && apt-get install -y git && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

WORKDIR /app

ENTRYPOINT [ "python", "run_hm.py" ]