import os
import random
import time
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional as F

from utils import utils
from utils.visualize import *

plt.rcParams["savefig.bbox"] = 'tight'


def main(args):
    print(args)
    device = torch.device('cuda')

    print("Creating model")
    kwargs = {
            "trainable_backbone_layers": args.trainable_backbone_layers
        }
    if "rcnn" in args.checkpoint:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(num_classes=2, num_keypoints=17,
                                                                **kwargs)
    model.load_state_dict(torch.load(args.checkpoint))                                                          
    model.to(device)
    model.eval()

    ###########################
    # Get random input dataset
    ###########################
    """
    Because change_gear / reach_side / standstill_or_waiting is empty in StateFarm dataset,
    this script will fail in 3/11 probailibity. Just re-run :)
    """
    path = os.path.join(args.input_path, random.choice(os.listdir(args.input_path)))
    random_img = random.choice(os.listdir(path))
    random_input = os.path.join(path, random_img)
    print(f'Selected image: {random_input}')

    image_int = torchvision.io.read_image(random_input)
    image = F.convert_image_dtype(torch.stack([image_int]), dtype=torch.float)

    ###########################
    # Get keypoint predictions
    ###########################
    model_time = time.time()
    outputs = model(image.to(device))
    
    # outputs[0] = {'boxes', 'labels', 'scores', 'keypoints', 'keypoints_scores'}
    outputs = [{k: v for k, v in t.items()} for t in outputs]
    model_time = time.time() - model_time
    print(f'Model time: {model_time}')
    
    ###########################
    # Get valid predictions
    ###########################
    score_threshold = 0.9
    keypoint_score_threshold = 0.8
    boxes = outputs[0]['boxes'][outputs[0]['scores'] > score_threshold]
    box_scores = outputs[0]['scores'][outputs[0]['scores'] > score_threshold]
    keypoints = outputs[0]['keypoints'][outputs[0]['scores'] > score_threshold]
    keypoints_scores = outputs[0]['keypoints_scores'][outputs[0]['scores'] > score_threshold]

    for i in range(keypoints.shape[0]):
        for j in range(keypoints.shape[1]):
            if keypoints_scores[i][j] < keypoint_score_threshold:
                for k in range(3):
                    keypoints[i][j][k] = 0

    ###########################
    # Draw bbox & kp
    ###########################
    box_labels = []
    for i in range(len(box_scores)):
        box_labels.append(f'{box_scores[i]}')

    result = torchvision.utils.draw_bounding_boxes(image=image_int, boxes=boxes, colors="green", width=4, labels=box_labels, font_size=12)
    result2 = draw_keypoints(image_int, keypoints)
    show(result, 'boxes_' + random_img + '.png')
    show(result2, 'kp_'+ random_img + '.png')

    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser = argparse.ArgumentParser(description='Evaluation for object detecion task (COCO dataset)')
    
    parser.add_argument('--input-path', default='/data/DMD-Driver-Monitoring-Dataset/test', help='Image path')
    parser.add_argument('--checkpoint', 
                        default='/home/esoc/psk/keypoint_dmd/keypointrcnn_resnet50_fpn_coco-fc266e95.pth',
                        help='model pth file path')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument("--mode", type=str, default='DMD', help='DMD, SF')
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                        help='number of trainable layers of backbone')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)