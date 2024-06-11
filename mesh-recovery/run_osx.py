import os
import sys
import argparse
import os.path as osp
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))

from config import cfg
import cv2

from common.base import Demoer
from common.utils.preprocessing import process_bbox, generate_patch_image
from common.utils.vis import render_mesh
from common.utils.human_models import smpl_x
from tqdm import tqdm

cudnn.benchmark = True

def main(args):
    # Load the YOLO detector and OSX model.
    osx_model = Demoer()
    osx_model._make_model()
    osx_model.model.eval()
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    frames = read_frames(args.input_path)
    inpaint_frames = read_frames(args.inpaint_path)

    print(f'Frames: {len(frames)}')
    print(f'Inpaint Frames: {len(inpaint_frames)}')

    final_frames = []
    assert len(frames) == len(inpaint_frames)
    for original_frame, inpaint_frame in tqdm(zip(frames, inpaint_frames), total=len(frames)):
        final_frame = process_frame(original_frame, inpaint_frame, osx_model, yolo)
        final_frames.append(final_frame)
    
    final_save_path = args.inpaint_path.replace('.mp4', '_final.mp4')
    save_frames(final_frames, final_save_path)


@torch.no_grad()
def process_frame(original_frame, inpaint_frame, osx_model, yolo):
    # Detect human BBOX with YOLOv5
    results = yolo(original_frame)
    person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]
    class_ids, confidences, boxes = [], [], []
    for detection in person_results:
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        class_ids.append(class_id)
        confidences.append(confidence)
        boxes.append([x1, y1, x2-x1, y2-y1])
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    h, w = original_frame.shape[:2]
    for index in indices:
        bbox = boxes[index]  # x,y,h,w
        bbox = process_bbox(bbox, w, h)
        img, _, _ = generate_patch_image(original_frame, bbox, 1.0, 0.0, False, cfg.input_img_shape)
        img = transforms.ToTensor()(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]
        inputs, targets, meta_info = {'img': img}, {}, {}

        # Mesh recovery
        out = osx_model.model(inputs, targets, meta_info, 'test')
        mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

        # Render mesh
        focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
        princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
        inpaint_frame = render_mesh(inpaint_frame, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
    
    return inpaint_frame

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, required=True, help='Input path to video')
    parser.add_argument('--inpaint_path', '-p', type=str, required=True, help='Input path to the inpainted video')
    parser.add_argument('--encoder-setting', '-es', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--decoder-setting', '-ds', type=str, default='normal', choices=['normal', 'wo_face_decoder', 'wo_decoder'])
    parser.add_argument('--ckpt', '-c', type=str, default='../pretrained_models/osx_l.pth.tar')

    args = parser.parse_args()
    cfg.set_args('0')
    cfg.set_additional_args(
        encoder_setting=args.encoder_setting,
        decoder_setting=args.decoder_setting,
        pretrained_model_path=args.ckpt
    )

    main(args)
