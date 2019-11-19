"""
Postprocessing module.
"""


from typing import List
from typing import Tuple
from typing import Union

import torch
import numpy as np
import torch.nn.functional as F


NdArray = np.ndarray

@torch.no_grad()
def heatmap_to_labeled_bboxes(heatmap: NdArray,
                              score_threshold: float = 0.1,
                             ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Convert heatmap tensor to labeled bounding boxes."""

    N = len(heatmap)

    values, indices = F.max_pool2d(
        heatmap[:, 0:-4, :, :],kernel_size=3, padding=1, stride=1, return_indices=True)

    batch_indices, labels, i_indices, j_indices = torch.where(
       ((values >= score_threshold) & (indices == 4)))

    w = heatmap[batch_indices, -4, i_indices, j_indices]
    h = heatmap[batch_indices, -3, i_indices, j_indices]
    dx = heatmap[batch_indices, -2, i_indices, j_indices]
    dy = heatmap[batch_indices, -1, i_indices, j_indices]
    scores = heatmap[batch_indices, labels, i_indices, j_indices]

    x1 = j_indices + dx - w / 2.
    y1 = i_indices + dy - h / 2.
    x2 = j_indices + dx + w / 2.
    y2 = i_indices + dy + h / 2.

    bboxes = torch.transpose(torch.stack([x1, y1, x2, y2]),1,0)

    if len(bboxes[0])==0:
        return

    bboxes_list = []
    labels_list = []
    scores_list = []
    for i in range(N):
        batch_mask = batch_indices == i
        if batch_mask.sum():
            bboxes_list.append(bboxes[batch_mask])
            labels_list.append(labels[batch_mask])
            scores_list.append(scores[batch_mask])

    return bboxes_list, labels_list, scores_list