# SMPLy Private: From Mask to Meshes in Action Recognition

Read the manuscript: [SMPLy Private](assets/SMPLy_Private.pdf)

### Overview
![Teaser Image](/assets/tennis.gif)

> In this paper, we introduce *Mask2Mesh* (M2M), a new framework that preserves privacy in action recognition by replacing real humans in videos with 3D meshes using the SMPL-X model. This method maintains the quality of human movement and expression details necessary for accurate recognition, unlike traditional privacy methods that degrade data quality. Empirical results demonstrate that our approach achieves performance within 0.5% of models trained on real video data.

![Overview of M2M](/assets/smply-p.png)

### Table Of Contents
1. [Human Masking & Inpainting](#human-masking--inpainting)
    - [Installation](#installation)
    - [Usage](#usage)
2. [Body Mesh Recovery](#body-mesh-recovery)
    - [Installation](#installation-1)
    - [Usage](#usage-1)
3. [Pre-training & Evaluation](#pre-training--evaluation)
    - [Installation](#installation-2)
    - [Usage](#usage-2)
        - [Pre-training](#pre-training)
        - [Supervised Alignment](#supervised-alignment)
        - [Downstream Evaluation](#downstream-evaluation)
    - [Configuration](#configuration)

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
Please follow the instructions in [`OSX/README.md`](https://github.com/IDEA-Research/OSX/blob/main/README.md) to setup the OSX codebase. We provide a pinned version of the dependencies in [`mesh-recovery/requirements.txt`](mesh-recovery/requirements.txt) to help installation. Post installation, our script, [`mesh-recovery/run_osx.py`](mesh-recovery/run_osx.py) can be copied to [`OSX/demo/run_osx.py`](https://github.com/IDEA-Research/OSX/tree/main/demo).

#### Usage
Ensure that you first run the human-masking and inpainting on the original video to obtain the inpainted video. The body mesh recovery algorithm can then be run to recover the body mesh from the original video, and then overlay it on the inpainted video using the following command:
```shell
python run_osx.py -i /path/to/original_video.mp4 -p /path/to/inpainted_video.mp4
```

### Pre-training & Evaluation
We use, for pre-training, a VideoMAE model and evaluate its performance on various downstream video action recognition datasets, such as UCF101, HMDB51, Mini-SSV2, Diving48, IkeaFA, and UAV-Human.

#### Data
Before running any training-based scripts, ensure you have downloaded the relevant datasets for downstream evaluation from the following table. We provide the exact splits used for training in [`training/data`](/training/data). Please refer to our [datasheet](/training/data/dfd.md) for more details.

| Downstream Data | Download Link |
|--------------------|-----------------------|
| UCF101             | [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) |
| HMDB51             | [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) |
| Diving48           | [Diving48](http://www.svcl.ucsd.edu/projects/resound/dataset.html) |
| IkeaFA             | [IkeaFA](https://tengdahan.github.io/ikea.html) |
| UAV-Human          | [UAV-Human](https://github.com/sutdcv/UAV-Human) |


Furthermore, you will need to preprocess the data such that the splits are as follows to run our code: 
```txt
/path/to/dataset/UCF101/
  ├── train/
  │   ├── video1.mp4
  │   ├── video2.mp4
  │   └── ...
  └── test/
      ├── video1.mp4
      ├── video2.mp4
      └── ...

/path/to/dataset/HMDB51/
  ├── train/
  │   ├── video1.mp4
  │   ├── video2.mp4
  │   └── ...
  └── test/
      ├── video1.mp4
      ├── video2.mp4
      └── ...

# And similarly for the other datasets...
```

#### Installation
You can install the required dependencies using pip:

```
transformers==4.39.0
accelerate==0.27.2
datasets==2.19.0
opencv-python==4.9.0.80
imageio[ffmpeg]==2.34.1
```

#### Usage
##### Pre-training
To pre-train the VideoMAE model, run:
```py
python training/videomae.py --data_dir /path/to/data --output_dir /path/to/output --stage pretrain
```

##### Supervised Alignment
To perform supervised alignment, run:
```py
python training/videomae.py --data_dir /path/to/data --output_dir /path/to/output --label_file /path/to/labels.txt --stage align
```
The label file should be of `.txt` format. An example is as follows: 
```txt
vid1.mp4,0
vid2.mp4,1
```

##### Downstream Evaluation
To evaluate the pre-trained model on a downstream task, run:
```py
python training/downstream_eval.py --data_dir /path/to/data --model_dir /path/to/pretrained/model --train_label_file /path/to/train_labels.txt --test_label_file /path/to/test_labels.txt --batch_size 8 --learning_rate 1e-4 --num_epochs 30 --warmup_epochs 5 --num_workers 2 --eval_interval 5 --eval_type fine_tune --dataset UCF101
```

#### Configuration
You can use our `config.yaml` file to set up your configurations for pre-training, alignment, and evaluation.
