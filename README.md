# pp-smpl

### Installation - HybrIK
Run the following block to download the HybrIK code-base.
```
git clone https://github.com/Jeff-sjtu/HybrIK.git
cp Dockerfile HyrbIK/Dockerfile
cd HybrIK
```

In the `setup.py` file, change `opencv-python==4.1.2.30` to `opencv-python-headless`. Next, install the required model-files from Google Drive.
```
pip install gdown

gdown https://drive.google.com/uc?id=1un9yAGlGjDooPwlnwFpJrbGHRiLaBNzV
unzip model_files.zip

mkdir pretrained_models
gdown https://drive.google.com/uc?id=1R0WbySXs_vceygKg_oWeLMNAZCEoCadG -O pretrained_models
```

Ensure that the current working directory is `HybrIK/` and the Dockerfile is in the current working directory. Furthermore, also ensure the model-files are available at `HybrIK/pretrained_models/*.pth` and `HybrIK/model_files/*`. Then we can build the Docker image.

By default, it will launch the bash shell.
```
docker build -t ppsmpl .
docker run --gpus all -it ppsmpl
```
Once the shell is launched, you can run the following:
```
python scripts/demo_video_x.py --video-name examples/dance.mp4 --out-dir res_dance --save-pt --save-img
```
