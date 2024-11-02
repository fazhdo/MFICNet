import torch
import torch.nn as nn

from ..builder import LOSSES

@LOSSES.register_module()
class EuclideanLoss_with_Uncertainty2(nn.Module):
    def __init__(self):
        super(EuclideanLoss_with_Uncertainty2, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)   # 求两组变量间的2范数

    def forward(self, pred, target, mask, certainty):
        loss_reg = self.pdist(pred, target)         # 对应的预测和真实坐标差值的二范数 [4 60 80]`
        certainty_map = torch.clamp(certainty, 1e-6)  # certainty_map 置信度[4 60 80] [0,1]
        loss_map = (3 * torch.log(certainty_map) + loss_reg.unsqueeze(dim=1) / (2 * certainty_map.pow(2))).squeeze(dim=1)

        loss_map = loss_map * mask # 深度大于0的才有损失
        loss =torch.sum(loss_map) / mask.sum()

        if mask is not None:
            valid_pixel = mask.sum() + 1
            diff_coord_map = mask * loss_reg

        thres_coord_map = torch.clamp(diff_coord_map - 0.05, 0)
        num_accurate = valid_pixel - thres_coord_map.nonzero().shape[0]
        accuracy = num_accurate / valid_pixel
        loss1 = torch.sum(loss_reg*mask) / mask.sum()
        return loss, accuracy
    
class EuclideanLoss_with_Uncertainty(nn.Module):
    def __init__(self):
        super(EuclideanLoss_with_Uncertainty, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)   # 求两组变量间的2范数

    def forward(self, pred, target, certainty):
        loss_reg = self.pdist(pred, target)         # 对应的预测和真实坐标差值的二范数 [4 60 80]`
        certainty_map = torch.clamp(certainty, 1e-6)  # certainty_map 置信度[4 60 80] [0,1]
        loss_map = (3 * torch.log(certainty_map) + loss_reg.unsqueeze(dim=2) / (2 * certainty_map.pow(2))).squeeze(dim=2)
        loss =torch.sum(loss_map) / (pred.shape[0] * pred.shape[1])

        valid_pixel = pred.shape[0] * pred.shape[1]
        diff_coord_map = loss_reg.reshape(-1, )
        thres_coord_map = torch.clamp(diff_coord_map - 0.05, 0)
        num_accurate = valid_pixel - thres_coord_map.nonzero().shape[0]
        accuracy = num_accurate / valid_pixel
        
        accuracy = torch.as_tensor(accuracy)
        return loss, accuracy

@LOSSES.register_module()
class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)   # 求两组变量间的2范数

    def forward(self, pred, target):
        b, n, c = pred.shape
        pred = pred.view(-1, c)
        target = target.view(-1, c)
        loss = self.pdist(pred, target)
        loss = torch.sum(loss, 0)
        loss /= pred.shape[0]
        import pdb; pdb.set_trace()
        return loss
    
@LOSSES.register_module()
class EuclideanLoss2(nn.Module):
    def __init__(self):
        super(EuclideanLoss2, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)   # 求两组变量间的2范数

    def forward(self, pred, target, mask):

        n, c, h, w = pred.size()
        pred = pred.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        pred = pred[mask.view(n * h * w, 1).repeat(1, c) == 1.]
        pred = pred.view(-1, c)

        target = target.transpose(1,2).transpose(2,3).contiguous().view(-1, c)
        target  = target[mask.view(n * h * w, 1).repeat(1, c) == 1.]
        target = target.view(-1,c)

        loss = self.pdist(pred, target)
        loss = torch.sum(loss, 0)
        loss /= mask.sum()
        return loss
    
@LOSSES.register_module()
class EuclideanLoss_EAAI(nn.Module):
    def __init__(self):
        super(EuclideanLoss_EAAI, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)   # 求两组变量间的2范数

    def forward(self, pred, target, mask, certainty):
        target = target.permute(0,2,3,1)
        pred = pred.permute(0,2,3,1)

        loss_reg = self.pdist(pred, target)
        certainty_map = torch.max(certainty.cuda(), torch.tensor(1e-6).cuda())
        loss_map = 3 * torch.log(certainty_map.squeeze(dim=1)) + loss_reg / (2 * certainty_map.squeeze(dim=1).pow(2))
        loss_map = loss_map * mask
        loss = torch.sum(loss_map) / mask.sum()

        if mask is not None:
            valid_pixel = mask.sum() + 1
            diff_coord_map = mask * loss_reg

        thres_coord_map = torch.max(diff_coord_map - 0.05, torch.tensor([0.]).cuda())
        num_accurate = valid_pixel - thres_coord_map.nonzero().shape[0]
        accuracy = num_accurate / valid_pixel
        loss1 = torch.sum(loss_reg * mask) / mask.sum()
        return loss, accuracy, loss1