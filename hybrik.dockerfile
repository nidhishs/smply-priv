FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

COPY hybrik/requirements.txt /app/requirements.txt

RUN conda install fvcore=0.1.5.post20221221 iopath=0.1.9 -c fvcore -c iopath -c conda-forge -y && \
    conda install pytorch3d=0.7.3 -c pytorch3d -y && \
    conda clean --all -y && \
    pip install -r /app/requirements.txt

COPY hybrik /app

WORKDIR /app

ENTRYPOINT [ "python", "run_hybrik.py" ]