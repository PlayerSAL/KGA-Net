import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_cls_loss(pred_dict, attr_label_tensor_dict, cfg):
    '''
    Compute classification loss
    :param pred_dict:
    :param attr_label_tensor_dict:
    :param cfg:
    :return:
    '''
    losses_dict = {}
    alpha = 0.1
    for (attr, weight) in zip(cfg.ATTR_LIST, cfg.ATTR_WEIGHT):
        gamma = ((torch.sum(1 / torch.FloatTensor(weight))) / len(weight)).to(cfg.MODEL.DEVICE)
        losses_dict[f"loss_{attr}"] = F.cross_entropy(
            pred_dict[attr], attr_label_tensor_dict[attr],
            ignore_index=-1,
            weight=torch.FloatTensor(weight).to(cfg.MODEL.DEVICE)
        ) * alpha * gamma
    return losses_dict



class CenterGramLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, cfg, num_classes=10, feat_dim=2, use_gpu=True, mode="2D"):
        super(CenterGramLoss, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.mode = mode

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, info_dict, attr_label_tensor_dict, file_names):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            attr_label_tensor_dict: ground truth labels with shape (batch_size).
        """
        indexes_2d = torch.tensor(["dataset_image" in f.lower() for f in file_names]).cuda()
        indexes_video = torch.tensor(["dataset_video" in f.lower() for f in file_names]).cuda()
        features = info_dict["features"]
        B, T, C = features.shape
        features = torch.reshape(features, (B * T, C))


        labels = attr_label_tensor_dict["Pathology"]
        batch_size = features.size(0)
        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(features, self.centers.t(), beta=1, alpha=-2)
        distmat = torch.reshape(distmat, (B, T, self.num_classes))
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels_busi = labels.unsqueeze(1).unsqueeze(1).expand(B, T, self.num_classes)
        indexes_2d = indexes_2d.unsqueeze(1).unsqueeze(1).expand(B, T, self.num_classes)
        mask = labels_busi.eq(classes.expand(B, T, self.num_classes)) * indexes_2d
        dist = distmat * mask.float()
        loss = (dist.clamp(min=1e-12, max=1e+12).sum() / (mask.sum() + 1e-5)) * self.cfg.CENTER_WEIGHT

        features = info_dict["features"]  # BTC
        B, T, C = features.shape
        features = torch.reshape(features, (B * T, C))
        attn = 1 - info_dict["attn"].squeeze(2)  # BT
        batch_size, C = features.size()
        center_agent = self.centers.detach()
        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(center_agent, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(features, center_agent.t(), beta=1, alpha=-2)
        distmat = torch.reshape(distmat, (B, T, self.num_classes))
        inds = labels.unsqueeze(1).unsqueeze(1).expand(B, T, 1)
        distmat = torch.gather(distmat, 2, inds).squeeze(2)  # BT
        normalized_attention = (attn - torch.mean(attn, dim=1).unsqueeze(1)) * indexes_video.unsqueeze(1)
        normalized_distance = (distmat - torch.mean(distmat, dim=1).unsqueeze(1)) + 1e-5
        normalized_distance = normalized_distance * indexes_video.unsqueeze(1) / \
                              torch.sqrt(torch.sum(normalized_distance ** 2, dim=1)).unsqueeze(1)
        cost = torch.nn.functional.mse_loss(
            torch.matmul(normalized_attention.unsqueeze(2), normalized_attention.unsqueeze(1)),
            torch.matmul(normalized_distance.unsqueeze(2), normalized_distance.unsqueeze(1)),
        ) * 100

        loss = loss + cost

        return loss, distmat
