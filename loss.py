import torch
import torch.nn.functional as F


def regr_loss(regr, gt_regr, mask):
    num = mask.float().sum()*2

    regr = regr[mask == 1]
    gt_regr = gt_regr[mask == 1]
    regr_loss = F.l1_loss(
        regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _neg_loss(pred, gt,alpha=2,beta=4):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, beta)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def criterion(prediction, true, size_average=True):

    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])

    mask_loss = _neg_loss(pred_mask[:,None,:,:],true[:,0:1,:,:])


    size_loss_x = regr_loss(prediction[:,-4,:,:],
                                    true[:,-4,:,:],true[:,0])
    size_loss_y = regr_loss(prediction[:,-3,:,:],
                                    true[:,-3,:,:],true[:,0])

    offset_loss_x = regr_loss(prediction[:,-2,:,:],
                                        true[:,-2,:,:],true[:,0])

    offset_loss_y = regr_loss(prediction[:,-1,:,:],
                                        true[:,-1,:,:],true[:,0])
          
    return mask_loss, size_loss_x+size_loss_y, offset_loss_x+offset_loss_y