import os
from typing import Union, Optional, List, Tuple, Text, BinaryIO
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageColor

import torch
import torchvision.transforms.functional as F


"""
"keypoints": [
            "nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle"
        ]
# Start with 1 (NOT 0)
"skeleton": [
            [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
            [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
        ]
"""

EDGES = [
    (13, 15),
    (11, 13),
    (14, 16),
    (12, 14),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6)
]

NUM_EDGES = len(EDGES)

COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0), \
        (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), \
        (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]

plt.rcParams["savefig.bbox"] = 'tight'

def show(imgs, output_name):
    #[C, H, W]
    height = imgs.shape[1]
    width = imgs.shape[2]
    dpi = 300
    figsize = width / float(dpi), height / float(dpi)

    #fig = plt.figure(figsize=figsize)
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    plt.savefig(os.path.join('./results', output_name), dpi=dpi)


@torch.no_grad()
def draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    alpha: float = 0.8,
    thickness: int = 2,
    font: Optional[str] = None
) -> torch.Tensor:
    """
    Draw keypoints & skeletons
    
    Args:
        image (Tensor): Tensor of shape (C x H x W) and dtype uint8.
        keypoints (Tensor): Tensor of keypoints (N x 17 x 3, dtype=float). [x, y, v] format, 17 keypoints in COCO format.
        alpha (float): Float number between 0 and 1 denoting the transparency of the masks.
            0 means full transparency, 1 means no transparency.

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size(0) not in {1, 3}:
        raise ValueError("Only grayscale and RGB images are supported")

    if image.size(0) == 1:
        image = torch.tile(image, (3, 1, 1))

    ndarr = image.permute(1, 2, 0).numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)

    #keypoints[box, 17, (x,y,v)]
    img_keypoints = keypoints.to(torch.int64).tolist()

    for j, kp in enumerate(img_keypoints): #box level
        for i in range(NUM_EDGES-1):
            #EDGES[i] = (k1, k2)
            if kp[EDGES[i][0]][2] == 0 or kp[EDGES[i][1]][2] == 0: #if v=0, pass
                continue
                
            x = (kp[EDGES[i][0]][0], kp[EDGES[i][1]][0])
            y = (kp[EDGES[i][0]][1], kp[EDGES[i][1]][1])

            draw.line((x[0], y[0], x[1], y[1]), fill=COLORS[i], width=3)

        for k in range(len(kp)): #keypoint level
            if kp[k][2] == 1:
                x = kp[k][0]
                y = kp[k][1]
                draw.ellipse([(x-thickness, y-thickness), (x+thickness, y+thickness)], fill=(0, 255, 0))
                draw.text(xy=(x+thickness+2, y+thickness+2), text=f'{k}', fill=(0,0,0))

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)
