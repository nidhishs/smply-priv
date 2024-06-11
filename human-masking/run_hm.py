import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from e2fgvi import run_e2fgvi, setup_e2fgvi
from tqdm import tqdm

MASKING_MODEL = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
E2FGVI_MODEL = "ckpt/E2FGVI-HQ-CVPR22.pth"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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


def get_human_masks(predictor, frames, progress=False):
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


def main(args):
    logger.info("Reading frames from: %s, resizing: %s", args.video, args.resize)
    frames = read_frames(args.video, args.resize)

    os.makedirs(args.output_dir, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(args.video))[0]

    if args.resize:
        resize_path = os.path.join(args.output_dir, f"{file_name}_resize.mp4")
        save_frames(frames, resize_path)
        logger.info("Saved resized video to: %s", resize_path)

    logger.info("Setting up human masking model: %s", MASKING_MODEL)
    masking_model = setup_model(MASKING_MODEL)
    logger.info("Setting up inpainting model: %s", E2FGVI_MODEL)
    e2fgvi_model = setup_e2fgvi(E2FGVI_MODEL)

    try:
        logger.info("Getting human masks...")
        masks = get_human_masks(masking_model, frames, progress=True)
    except Exception as e:
        logger.error("Failed to get human masks: %s", e)
        return

    try:
        logger.info("Inpainting frames...")
        inpainted_frames = run_e2fgvi(e2fgvi_model, frames, masks, progress=True)
    except Exception as e:
        logger.error("Failed to inpaint: %s", e)
        return
    
    inpaint_path = os.path.join(args.output_dir, f"{file_name}_inpaint.mp4")
    save_frames(inpainted_frames, inpaint_path)
    logger.info("Saved inpainted video to: %s", inpaint_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mask and in-paint humans in a given video.")
    parser.add_argument("--video", "-v", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output-dir", "-o", type=str, default="output", help="Output directory to save the inpainted video")
    parser.add_argument("--resize", "-r", type=int, nargs=2, help="Resize the video to the given width and height")

    args = parser.parse_args()

    main(args)