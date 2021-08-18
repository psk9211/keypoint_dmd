import datetime
import os
import time

import torch
import torch.utils.data
import torchvision

from utils.coco_utils import get_coco, get_coco_kp

from utils.engine import evaluate

from utils import presets
from utils import utils


def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    return presets.DetectionPresetTrain() if train else presets.DetectionPresetEval()


def main(args):
    print(args)
    device = torch.device('cuda')

    # Data loading code
    print("Loading data")
    dataset_test, num_classes = get_dataset(args.dataset, "val", get_transform(train=False), args.data_path)

    print("Creating data loaders")
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    kwargs = {
            "trainable_backbone_layers": args.trainable_backbone_layers
        }
    if "rcnn" in args.checkpoint:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(num_classes=num_classes, num_keypoints=17,
                                                                **kwargs)
    model.load_state_dict(torch.load(args.checkpoint))                                                          
    model.to(device)
    model.eval()

    evaluate(model, data_loader_test, device=device)
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser = argparse.ArgumentParser(description='Evaluation for object detecion task (COCO dataset)')
    
    parser.add_argument('--data-path', default='/data/coco2017', help='dataset')
    parser.add_argument('--dataset', default='coco_kp', help='dataset')
    parser.add_argument('--checkpoint', 
                        default='/home/esoc/psk/keypoint_dmd/keypointrcnn_resnet50_fpn_coco-fc266e95.pth',
                        help='model pth file path')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument("--type", type=str, default='pth')
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )


    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)