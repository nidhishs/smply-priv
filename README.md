# pp-smpl

### HybrIK
Install the required model-files from Google Drive. Ensure the model-files are available at `hybrik/pretrained_models/*.pth` and `hybrik/model_files/*`. 
```shell
pip install gdown

mkdir -p hybrik/pretrained_models
mkdir -p hybrik/model_files

gdown https://drive.google.com/uc?id=1un9yAGlGjDooPwlnwFpJrbGHRiLaBNzV
gdown https://drive.google.com/uc?id=1R0WbySXs_vceygKg_oWeLMNAZCEoCadG -O hybrik/pretrained_models/hybrikx_rle_hrnet.pth
unzip model_files.zip -d hybrik
```

From the root directory, execute the following Docker commands to first build, and then launch the container.
```shell
docker build -t ppsmpl-hybrik -f hybrik.dockerfile .
docker run --gpus all -v $(pwd)/hybrik/pretrained_models:/app/pretrained_models -v $(pwd)/hybrik/model_files:/app/model_files -v $(pwd)/hybrik/output:/app/output -it ppsmpl-hybrik --video-name <path-to-video-file>.mp4 --out-dir output --save-pk --save-img
```

### Human Masking
We use Detectron2 for masking humans in videos. From the root directory, execute the following Docker commands to first build, and then launch the container.
```shell
docker build -t ppsmpl-hm -f hm.dockerfile .
docker run --gpus all -it ppsmpl-hm -i <input-video-path>.mp4 -o <output-video-path>.mp4
```
