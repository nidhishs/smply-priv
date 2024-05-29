import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import glob
import logging
import time

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from e2fgvi import run_e2fgvi, setup_e2fgvi
from tqdm import tqdm

MASKING_MODEL = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
E2FGVI_MODEL = "ckpt/E2FGVI-HQ-CVPR22.pth"

os.makedirs('logs', exist_ok=True)
log_format = "%(asctime)s [ %(levelname)8s ] %(message)s"
logger = logging.getLogger('run-hm')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join('logs', f"run-hm-{int(time.time())}.log"))
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(log_format)
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

def setup_model(config_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    return predictor


def read_frames(video_path, size=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if size:
            frame = cv2.resize(frame, size)
        frames.append(frame)
    cap.release()
    frames = [np.array(f, dtype=np.uint8) for f in frames]
    return frames


def save_frames(frames, output_path, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    size = frames[0].shape[1], frames[0].shape[0]
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    for frame in frames:
        out.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    out.release()


def get_masks(predictor, frames, progress=False):
    masks = []
    for frame in tqdm(frames) if progress else frames:
        instances = predictor(frame)["instances"]
        # Filter out only the person class (class ID 0)
        person_masks = instances.pred_masks[instances.pred_classes == 0].cpu().numpy()
        combined_mask = (np.sum(person_masks, axis=0) > 0).astype(np.uint8)
        dilated_mask = (
            cv2.dilate(
                combined_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=4,
            )
            * 255
        )
        masks.append(dilated_mask)
    return masks

def process_video(masking_model, video_path, input_dir, output_dir, inpaint_method=None, inpaint_model=None, resize=False):
    frames = read_frames(video_path, args.resize)
    if resize:
        resized_path = os.path.join(
            output_dir, 'resized', os.path.relpath(video_path, input_dir)
        )
        os.makedirs(os.path.dirname(resized_path), exist_ok=True)
        save_frames(frames, resized_path)
        logger.info("Saved resized video to: %s", resized_path)
        
    logger.info("Getting human masks...")
    masks = get_masks(masking_model, frames, progress=True)
    mask_path = os.path.join(
        output_dir, 'masks',
        os.path.relpath(video_path, input_dir).replace(".mp4", "_masks.mp4")
    )
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    save_frames(masks, mask_path)
    logger.info("Saved masks video to: %s", mask_path)
    
    if not inpaint_method:
        return
    
    logger.info("Inpainting...")
    if inpaint_method == "e2fgvi":
        assert inpaint_model is not None, "E2FGVI model not loaded"
        inpainted_frames = run_e2fgvi(
            inpaint_model, frames, masks, progress=True
        )
    elif inpaint_method == "ns":
        inpainted_frames = []
        for frame, mask in zip(frames, masks):
            inpainted_frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)
            inpainted_frames.append(inpainted_frame)

    assert inpainted_frames, "No inpainted frames found"
    inpaint_path = os.path.join(
        output_dir, 'inpainted',
        os.path.relpath(video_path, input_dir).replace(".mp4", f"_{inpaint_method}.mp4")
    )
    os.makedirs(os.path.dirname(inpaint_path), exist_ok=True)
    save_frames(inpainted_frames, inpaint_path)
    logger.info("Saved inpainted video to: %s", inpaint_path)


def main(args):
    video_paths = glob.glob(os.path.join(args.input_dir, "**/*.mp4"), recursive=True)
    logger.info("Reading videos from: %s. Found %d videos.", args.input_dir, len(video_paths))

    masking_model = setup_model(MASKING_MODEL)
    logger.info("Using masking-model: %s", MASKING_MODEL)
    
    if args.inpaint == "e2fgvi":
        logger.info("Inpainting using E2FGVI model: %s", E2FGVI_MODEL)
        inpaint_model = setup_e2fgvi(E2FGVI_MODEL, "cuda" if torch.cuda.is_available() else "cpu")
    elif args.inpaint == "ns":
        logger.info("Inpainting using OpenCV's Navier-Stokes method")
        inpaint_model = None

    tic = time.time()
    for i, video_path in enumerate(video_paths):
        logger.info("Processing video (%d/%d): %s", i + 1, len(video_paths), video_path)
        try:
            process_video(masking_model, video_path, args.input_dir, args.output_dir, args.inpaint, inpaint_model, args.resize)
        except Exception as e:
            logger.error("Error processing video: %s", str(e))
            
    toc = time.time()

    h, m, s = int(toc - tic) // 3600, int(toc - tic) // 60, int(toc - tic) % 60
    logger.info("Finished processing %d videos in %dh %dm %ds", len(video_paths), h, m, s)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mask and in-paint humans in a given video.")
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="Path to directory containing input videos.")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Path to directory to save output videos.")
    parser.add_argument("--inpaint", "-p", type=str, choices=["e2fgvi", "ns"], default="e2fgvi", help="Inpainting method")
    parser.add_argument("--resize", "-r", type=int, nargs=2, help="Resize the video to the given width and height")

    args = parser.parse_args()

    main(args)
