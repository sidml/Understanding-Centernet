import torch
import torch.nn as nn
from utils import _gather_feat, _tranpose_and_gather_feat

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100,model_scale=4):
    batch, cat, height, width = heat.size()

    heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    xs_raw = xs.view(batch, K, 1) + 0.5
    ys_raw = ys.view(batch, K, 1) + 0.5

    if reg is not None:
      reg = _tranpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1] # check if it's correct and not reversed with ys
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    

    wh = _tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
      wh = wh.view(batch, K, cat, 2)
      clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
      wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
      wh = wh.view(batch, K, 2)

    xs,ys = xs*model_scale,ys*model_scale

    xs_raw,ys_raw = xs_raw*model_scale,ys_raw*model_scale

    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - wh[..., 1:2] / 2, 
                    ys - wh[..., 0:1] / 2,
                    xs + wh[..., 1:2] / 2, 
                    ys + wh[..., 0:1] / 2], dim=2)

    bboxes_raw = torch.cat([xs_raw - wh[..., 1:2] / 2, 
                ys_raw - wh[..., 0:1] / 2,
                xs_raw + wh[..., 1:2] / 2, 
                ys_raw + wh[..., 0:1] / 2], dim=2)
      
    return bboxes_raw,bboxes, scores, clses