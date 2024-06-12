# SMPLy Private: From Mask to Meshes in Action Recognition

### Overview
![Teaser Image](/assets/tennis.gif)

> In this paper, we introduce *Mask2Mesh* (M2M), a new framework that preserves privacy in action recognition by replacing real humans in videos with 3D meshes using the SMPL-X model. This method maintains the quality of human movement and expression details necessary for accurate recognition, unlike traditional privacy methods that degrade data quality. Empirical results demonstrate that our approach achieves performance within $0.5%$ of models trained on real video data.

![Overview of M2M](/assets/smply-p.png)

### Table Of Contents
1. [Human Masking & Inpainting](#human-masking--inpainting)
    - [Installation](#installation)
    - [Usage](#usage)
2. [Body Mesh Recovery](#body-mesh-recovery)
    - [Installation](#installation-1)
    - [Usage](#usage-1)

### Human Masking & Inpainting
We use Detectron2's implemention of MaskRCNN w/ a ResNet-101 backbone for masking humans in videos and [E^2FGVI](https://github.com/MCG-NKU/E2FGVI) for inpainting.

#### Installation
Ensure the E^2FGVI model-files are available at `human-masking/ckpt/*.pth`.
```shell
pip install gdown

mkdir -p human-masking/ckpt
gdown https://drive.google.com/uc?id=10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3 -O human-masking/ckpt/E2FGVI-HQ-CVPR22.pth
```
You can install the required dependencies using pip:
```
opencv-python==4.9.0.80
detectron2 @ git+https://github.com/facebookresearch/detectron2.git@79f9147
mmcv==2.2.0
mmengine==0.10.4
```

#### Usage
To run the masking and inpainting, run the following command. The resulting videos will be available in `/output` by default.
```shell
python run_hm.py -v /path/to/input_video.mp4 -o /path/to/output_dir
```

You may also use the Dockerfile. From the `human-masking` directory, execute the following Docker commands to first build, and then launch the container.
```shell
docker build -t smplyp-hm .
docker run --gpus all -v $(pwd)/ckpt:/app/ckpt -v $(pwd)/output:/app/output -it smplyp-hm -v /path/to/input_video.mp4
```

### Body Mesh Recovery.
We use the [OSX](https://github.com/IDEA-Research/OSX) algorithm for body mesh recovery. Please refer to their [paper](http://arxiv.org/abs/2303.16160) for more details.

#### Installation
We depend on the OSX codebase for body mesh recovery. Please follow the instructions in [`OSX/README.md`](https://github.com/IDEA-Research/OSX/blob/main/README.md) to setup the OSX codebase. We provide a pinned version of the dependencies in [`mesh-recovery/requirements.txt`](mesh-recovery/requirements.txt) to help installation.

Post installation, our script, [`mesh-recovery/run_osx.py`](mesh-recovery/run_osx.py) can be copied to [`OSX/demo/run_osx.py`](https://github.com/IDEA-Research/OSX/tree/main/demo).

#### Usage
Ensure that you first run the human-masking and inpainting on the original video to obtain the inpainted video. The body mesh recovery algorithm can then be run to recover the body mesh from the original video, and then overlay it on the inpainted video using the following command:
```shell
python run_osx.py -i /path/to/original_video.mp4 -p /path/to/inpainted_video.mp4
```

### Pre-training & Evaluation
#### Data
We provide the exact splits used for training in [`training/data`](/training/data). Please refer to our [datasheet](/training/dfd.md) for more details.