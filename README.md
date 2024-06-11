# pp-smpl

### Human Masking & Inpainting
We use Detectron2 for masking humans in videos and E^2FGVI for inpainting. Ensure the E^2FGVI model-files are available at `human-masking/ckpt/*.pth`.
```shell
pip install gdown

mkdir -p human-masking/ckpt
gdown https://drive.google.com/uc?id=10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3 -O human-masking/ckpt/E2FGVI-HQ-CVPR22.pth
```
From the `human-masking` directory, execute the following Docker commands to first build, and then launch the container.
```shell
docker build -t ppsmpl-hm .
docker run --gpus all -v $(pwd)/ckpt:/app/ckpt -v $(pwd)/output:/app/output -it ppsmpl-hm -v INPUT_PATH
```

### Body Mesh Recovery.
We depend on the OSX codebase for body mesh recovery. Please follow the instructions in the [`OSX/README.md`](https://github.com/IDEA-Research/OSX/blob/main/README.md) to setup the OSX codebase. We provide a pinned version of the dependencies in [`mesh-recovery/requirements.txt`](mesh-recovery/requirements.txt) to help installation.

The file [`mesh-recovery/run_osx.py`](mesh-recovery/run_osx.py) can be copied to [`OSX/demo/run_osx.py`](https://github.com/IDEA-Research/OSX/tree/main/demo) to recover the body mesh from the original video, and then overlay it on the inpainted video.
```shell
python3 run_osx.py -i INPUT_VIDEO -p INPAINT_VIDEO
```