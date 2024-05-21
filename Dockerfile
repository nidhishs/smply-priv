FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN conda install fvcore=0.1.5.post20221221 iopath=0.1.9 -c fvcore -c iopath -c conda-forge -y
RUN conda install pytorch3d=0.7.3 -c pytorch3d -y

COPY . /app

WORKDIR /app
RUN pip install -e .

CMD ["/bin/bash"]
