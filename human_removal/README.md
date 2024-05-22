# Human Removal from Video

This part of the repository contains a script to remove humans from a video using Detectron2 instance segmentation.

## Setup

1. **Clone the repository**

    ```bash
    git clone https://github.com/nidhishs/human_removal.git
    cd human_removal
    ```

2. **Create and activate the conda environment**

    ```bash
    conda env create -f environment.yml
    conda activate human_removal
    ```

## Usage

Run the `human_removal.py` script with the input and output video paths as arguments.

```bash
python human_removal.py path_to_input_video.mp4 path_to_output_video.mp4
