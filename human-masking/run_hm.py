import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from tqdm import tqdm

MASKING_MODEL = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
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


def get_human_masks(frames, config_path, progress=False):
    predictor = setup_model(config_path)
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
    logger.info("Reading frames from: %s, resizing: %s", args.input_path, args.resize)
    frames = read_frames(args.input_path, args.resize)
    
    logger.info("Getting human masks using model: %s", MASKING_MODEL)
    masks = get_human_masks(frames, MASKING_MODEL, progress=True)
    
    mask_path = args.input_path.replace(".mp4", "_masks.mp4")
    save_frames(masks, mask_path)
    logger.info("Saved masks video to: %s", mask_path)
    
    if not args.inpaint:
        return

    if args.inpaint == "e2fgvi":
        from e2fgvi import run_e2fgvi
        logger.info("Inpainting using E2FGVI model: %s", E2FGVI_MODEL)
        inpainted_frames = run_e2fgvi(
            frames, masks, ckpt_path=E2FGVI_MODEL, progress=True
        )
    elif args.inpaint == "ns":
        logger.info("Inpainting using OpenCV's Navier-Stokes method")
        inpainted_frames = []
        for frame, mask in zip(frames, masks):
            inpainted_frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)
            inpainted_frames.append(inpainted_frame)
    else:
        raise ValueError("Invalid inpainting method")
    
    inpaint_path = args.input_path.replace(".mp4", f"_{args.inpaint}.mp4")
    save_frames(inpainted_frames, inpaint_path)
    logger.info("Saved inpainted video to: %s", inpaint_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mask and in-paint humans in a given video.")
    parser.add_argument("--input_path", "-i", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--inpaint", "-p", type=str, choices=["e2fgvi", "ns"], default="e2fgvi", help="Inpainting method")
    parser.add_argument("--resize", "-r", type=int, nargs=2, help="Resize the video to the given width and height")

    args = parser.parse_args()

    main(args)
