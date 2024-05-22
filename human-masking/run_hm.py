import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def setup_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    predictor = DefaultPredictor(cfg)
    return predictor


def process_frame(frame, predictor):
    outputs = predictor(frame)
    instances = outputs["instances"]

    # Filter out only the person class (class ID 0)
    person_masks = instances.pred_masks[instances.pred_classes == 0].cpu().numpy()

    for mask in person_masks:
        frame[mask] = (0, 0, 0)

    return frame


def process_video(input_video_path, output_video_path, predictor):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4)))
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame, predictor)
        out.write(processed_frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove humans from a video using Detectron2."
    )
    parser.add_argument("--input_path", "-i", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output_path", "-o", type=str, required=True, help="Path to the output video file")
    args = parser.parse_args()

    predictor = setup_predictor()
    process_video(args.input_path, args.output_path, predictor)
