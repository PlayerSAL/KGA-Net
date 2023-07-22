from itertools import chain
import copy

import torch
import torchvision
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.utils import comm
from torch import nn

from modeling.losses import compute_cls_loss
from utils.image_list import imagelist_from_tensors
from modeling.losses import CenterGramLoss as CenterLoss
from modeling.losses import ThyroidCenterGramLoss as TCenterLoss


# I3DNet Wrapper
@META_ARCH_REGISTRY.register()
class NetTemporalFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        feature_dim = cfg.MODEL.FEATURE_DIM
        if self.cfg.ORGAN == 'thyroid':
            self.attr_list = self.cfg.ATTR_LIST + self.cfg.ATTR_MULTI_LIST + \
                             self.cfg.SECOND_ATTR_LIST + self.cfg.SECOND_ATTR_MULTI_LIST
            self.center_loss = TCenterLoss(cfg=self.cfg, num_classes=2, feat_dim=feature_dim, mode=cfg.MODEL.MODE)
        elif self.cfg.ORGAN == 'breast':
            # ordinal regression 分为 5 个等级
            # 修改，增加了二级属性
            self.attr_list = [*self.cfg.ATTR_LIST, *self.cfg.SECOND_ATTR_MULTI_LIST]
            self.center_loss = CenterLoss(cfg=self.cfg, num_classes=2, feat_dim=feature_dim, mode=cfg.MODEL.MODE)
        else:
            raise ValueError(f"{self.cfg.ORGAN} not supported yet!")
        self.iter = 0

        self.net = Net2DCore(cfg)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8)
        # self.temporal_merger = nn.TransformerEncoder(encoder_layer, num_layers=6)
        # self.temporal_merger = nn.GRU(input_size=feature_dim, num_layers=3,
        #                               hidden_size=feature_dim//2, batch_first=True,
        #                               bidirectional=True)
        # self.weights_init(self.temporal_merger)

        self.attn = nn.Sequential(nn.Linear(feature_dim, 1),
                                  nn.Sigmoid())
        self.act = nn.Softmax(dim=1)
        if self.cfg.ORGAN == 'thyroid':
            for id, attr in enumerate(cfg.ATTR_LIST):
                self.add_module(f"attr_head_{id}", nn.Linear(feature_dim, len(ATTR_LIB[attr])))
            for id, attr in enumerate(cfg.ATTR_MULTI_LIST):
                self.add_module(f"attr_multi_head_{id}", nn.Linear(feature_dim, 2))
            for id, attr in enumerate(cfg.SECOND_ATTR_LIST):
                self.add_module(f"second_attr_head_{id}", nn.Linear(feature_dim, len(ATTR_LIB[attr])))
            for id, attr in enumerate(cfg.SECOND_ATTR_MULTI_LIST):
                self.add_module(f"second_attr_multi_head_{id}", nn.Linear(feature_dim, 2))

        elif self.cfg.ORGAN == 'breast':
            totaldim = 0
            for id, attr in enumerate(cfg.ATTR_LIST):
                self.add_module(f"attr_head_{id}", nn.Linear(feature_dim, len(ATTR_LIB_BREAST[attr])))
                totaldim += len(ATTR_LIB_BREAST[attr])
            # 增加了二级属性
            for id, attr in enumerate(cfg.SECOND_ATTR_MULTI_LIST):
                self.add_module(f"attr_head_{id + len(cfg.ATTR_LIST)}", nn.Linear(feature_dim, 2))
                totaldim += 2
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.to(self.device)

    @staticmethod
    def weights_init(x):
        if isinstance(x, torch.nn.GRU):
            for n, p in x.named_parameters():
                if 'weight_ih' in n:
                    for ih in p.chunk(3, 0):
                        torch.nn.init.xavier_uniform_(ih)
                elif 'weight_hh' in n:
                    for hh in p.chunk(3, 0):
                        torch.nn.init.orthogonal_(hh)
                elif 'bias_ih' in n:
                    torch.nn.init.zeros_(p)
                # elif 'bias_hh' in n:
                #     torch.nn.init.ones_(p)

        elif isinstance(x, torch.nn.GRUCell):
            for hh, ih in zip(x.weight_hh.chunk(3, 0), x.weight_ih.chunk(3, 0)):
                torch.nn.init.orthogonal_(hh)
                torch.nn.init.xavier_uniform_(ih)

            torch.nn.init.zeros_(x.bias_ih)

    def forward(self, inputs):
        if self.training:
            return self.forward_train(inputs)
        else:
            return self.forward_infer(inputs)

    def forward_train(self, batched_inputs):
        batchsize = len(batched_inputs)
        attr_label_tensor_dict = {}
        for attr in batched_inputs[0][0]['attr_label']:
            attr_label_tensor_dict[attr] = torch.LongTensor([x[0]['attr_label'][attr] for x in batched_inputs]).to(
                self.device)

        file_names = [b[0]["file_name"] for b in batched_inputs]
        batched_inputs = list(chain.from_iterable(batched_inputs))
        images = self.preprocess_image(batched_inputs, batchsize)
        images_tensor = torch.permute(images.tensor, (0, 2, 1, 3, 4))
        B, T, C, H, W = images_tensor.shape
        images_tensor = torch.reshape(images_tensor, (B * T, C, H, W))
        pred_list, features = self.net(images_tensor)
        pred_tuple = tuple(pred_list)
        aux_dict = self.postprocess(pred_tuple)

        #  Auxiliary loss, 增加每一帧的监督
        flatten_attr_label_tensor_dict = {}
        for key in attr_label_tensor_dict:
            label = copy.deepcopy(attr_label_tensor_dict[key])
            flatten_attr_label_tensor_dict[key] = label.unsqueeze(1).expand(B, T).reshape(B * T)
        if self.cfg.ORGAN == 'thyroid':
            aux_loss = compute_i3d_loss(aux_dict, flatten_attr_label_tensor_dict, self.cfg)["loss_病理"]
        else:
            aux_loss = compute_i3d_loss(aux_dict, flatten_attr_label_tensor_dict, self.cfg)["loss_Pathology"]


        if self.cfg.MODEL.MODE == "HYBRID":
            # self.net.requires_grad_(False)
            # self.center_loss.requires_grad_(False)
            # features = torch.reshape(features, (B, T, c))
            # h0 = torch.randn((2, B, self.temporal_merger.hidden_size)).cuda()

            # features = torch.reshape(features, (B, T, -1))
            # features_tm = self.temporal_merger(features)[0]
            # features = torch.reshape(features_tm, (B*T, -1))

            b, c = features.shape
            attn = self.attn(features)
            attn = torch.reshape(attn, (B, T, 1))
            features = torch.reshape(features, (B, T, c))
            features_temporal = features * attn
            features_temporal = torch.mean(features_temporal, dim=1)
            pred_list = []
            if self.cfg.ORGAN == 'thyroid':
                for id, attr in enumerate(self.cfg.ATTR_LIST):
                    pred_list.append(getattr(self, f"attr_head_{id}")(features_temporal))
                for id, attr in enumerate(self.cfg.ATTR_MULTI_LIST):
                    pred_list.append(getattr(self, f"attr_multi_head_{id}")(features_temporal))
                for id, attr in enumerate(self.cfg.SECOND_ATTR_LIST):
                    pred_list.append(getattr(self, f"second_attr_head_{id}")(features_temporal))
                for id, attr in enumerate(self.cfg.SECOND_ATTR_MULTI_LIST):
                    pred_list.append(getattr(self, f"second_attr_multi_head_{id}")(features_temporal))
            elif self.cfg.ORGAN == 'breast':
                for id, attr in enumerate(self.cfg.ATTR_LIST):
                    pred_list.append(getattr(self, f"attr_head_{id}")(features_temporal))
                # 增加了二级属性
                for id, attr in enumerate(self.cfg.SECOND_ATTR_MULTI_LIST):
                    pred_list.append(getattr(self, f"attr_head_{id + len(self.cfg.ATTR_LIST)}")(features_temporal))

        pred_tuple = tuple(pred_list)
        pred_dict = self.postprocess(pred_tuple)
        if self.cfg.MODEL.MODE == "HYBRID":
            pred_dict["attn"] = attn
        pred_dict["features"] = features
        losses_dict = compute_i3d_loss(pred_dict, attr_label_tensor_dict, self.cfg)
        loss_center, _ = self.center_loss(pred_dict, attr_label_tensor_dict, file_names)
        losses_dict["center"] = loss_center
        losses_dict["aux"] = aux_loss
        self.iter += 1
        return losses_dict

    def forward_infer(self, batched_inputs):
        batched_inputs = [batched_inputs]
        attr_label_tensor_dict = {}
        for attr in batched_inputs[0][0]['attr_label']:
            attr_label_tensor_dict[attr] = torch.LongTensor([x[0]['attr_label'][attr] for x in batched_inputs]).to(
                self.device)

        file_names = [b[0]["file_name"] for b in batched_inputs]
        batchsize = len(batched_inputs)
        batched_inputs = list(chain.from_iterable(batched_inputs))
        images = self.preprocess_image(batched_inputs, batchsize)
        images_tensor = torch.permute(images.tensor, (0, 2, 1, 3, 4))
        B, T, C, H, W = images_tensor.shape
        images_tensor = torch.reshape(images_tensor, (B * T, C, H, W))
        pred, features = self.net(images_tensor)

        if self.cfg.MODEL.MODE == "HYBRID":
            # features = torch.reshape(features, (B, T, -1))
            # features_tm = self.temporal_merger(features)[0]
            # features = torch.reshape(features_tm, (B * T, -1))

            b, c = features.shape
            # features = torch.reshape(features, (B, T, c))
            # features_tm = self.temporal_merger(features)[0]
            # features_tm = torch.reshape(features_tm, (B * T, -1))
            attn = self.attn(features)
            attn = torch.reshape(attn, (B, T, 1))
            features = torch.reshape(features, (B, T, c))
            features_temporal = features * attn
            features_temporal = torch.mean(features_temporal, dim=1)
            pred = []
            if self.cfg.ORGAN == 'thyroid':
                for id, attr in enumerate(self.cfg.ATTR_LIST):
                    pred.append(getattr(self, f"attr_head_{id}")(features_temporal))
                for id, attr in enumerate(self.cfg.ATTR_MULTI_LIST):
                    pred.append(getattr(self, f"attr_multi_head_{id}")(features_temporal))
                for id, attr in enumerate(self.cfg.SECOND_ATTR_LIST):
                    pred.append(getattr(self, f"second_attr_head_{id}")(features_temporal))
                for id, attr in enumerate(self.cfg.SECOND_ATTR_MULTI_LIST):
                    pred.append(getattr(self, f"second_attr_multi_head_{id}")(features_temporal))
            elif self.cfg.ORGAN == 'breast':
                for id, attr in enumerate(self.cfg.ATTR_LIST):
                    pred.append(getattr(self, f"attr_head_{id}")(features_temporal))
                # 增加了二级属性
                for id, attr in enumerate(self.cfg.SECOND_ATTR_MULTI_LIST):
                    pred.append(getattr(self, f"attr_head_{id + len(self.cfg.ATTR_LIST)}")(features_temporal ))

        pred_dict = self.postprocess(pred, is_train=False)

        if self.cfg.MODEL.MODE == "HYBRID":
            #
            pred_dict["attn"] = attn
            pred_dict["features"] = features
        _, distmat = self.center_loss(pred_dict, attr_label_tensor_dict, file_names)
        pred_dict["distmat"] = distmat
        return pred_dict

    def reset(self):
        """
        Reset caches to inference on a new video.
        """
        ...

    def preprocess_image(self, batched_inputs, batchsize):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device, non_blocking=True) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = imagelist_from_tensors(images, batchsize)
        return images

    def postprocess(self, pred, is_train=True):
        """
        Postprocess pred of I3DNetCore
        """
        pred_dict = {}
        # init里面没有初始化trt_on的属性
        # if not self.trt_on:
            # pred_tuple
        assert len(self.attr_list) == len(pred)
        for attr, res in zip(self.attr_list, pred):
            if is_train:
                pred_dict[attr] = res
            else:
                pred_dict[attr] = self.act(res)
        # else:
        #     assert len(self.attr_list) == len(pred)
        #     for attr in self.attr_list:
        #         pred_dict[attr] = self.act(pred[attr])

        return pred_dict


# I3DNet Core Module
class Net2DCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.backbone = torchvision.models.__dict__[cfg.MODEL.BACKBONE](pretrained=True)
        # output feature vector rather than kinetics classification head
        last_module, last_name = self._get_models_last(self.backbone)
        if isinstance(last_module, nn.Linear):
            self.backbone._modules[last_name] = nn.Sequential(nn.Identity(),
                                                              nn.Sigmoid())
        elif isinstance(last_module, nn.Sequential):
            seq_last_module, seq_last_name = self._get_models_last(last_module)
            last_module._modules[seq_last_name] = nn.Sequential(nn.Identity(),
                                                              nn.Sigmoid())

        # syncbn for eval & test consistency
        if comm.get_world_size() > 1:
            self.backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)

        feature_dim = cfg.MODEL.FEATURE_DIM
        if self.cfg.ORGAN == 'thyroid':
            for id, attr in enumerate(cfg.ATTR_LIST):
                self.add_module(f"attr_head_{id}", nn.Linear(feature_dim, len(ATTR_LIB[attr])))
            for id, attr in enumerate(cfg.ATTR_MULTI_LIST):
                self.add_module(f"attr_multi_head_{id}", nn.Linear(feature_dim, 2))
            for id, attr in enumerate(cfg.SECOND_ATTR_LIST):
                self.add_module(f"second_attr_head_{id}", nn.Linear(feature_dim, len(ATTR_LIB[attr])))
            for id, attr in enumerate(cfg.SECOND_ATTR_MULTI_LIST):
                self.add_module(f"second_attr_multi_head_{id}", nn.Linear(feature_dim, 2))

        elif self.cfg.ORGAN == 'breast':
            totaldim = 0
            for id, attr in enumerate(cfg.ATTR_LIST):
                self.add_module(f"attr_head_{id}", nn.Linear(feature_dim, len(ATTR_LIB_BREAST[attr])))
                totaldim += len(ATTR_LIB_BREAST[attr])
            # 增加了二级属性
            for id, attr in enumerate(cfg.SECOND_ATTR_MULTI_LIST):
                self.add_module(f"attr_head_{id + len(cfg.ATTR_LIST)}", nn.Linear(feature_dim, 2))
                totaldim += 2

    def forward(self, x):
        '''
        :param x: Tensor
        :return: tuple of Tensor
        '''
        features = self.backbone(x)

        pred_list = []
        if self.cfg.ORGAN == 'thyroid':
            for id, attr in enumerate(self.cfg.ATTR_LIST):
                pred_list.append(getattr(self, f"attr_head_{id}")(features))
            for id, attr in enumerate(self.cfg.ATTR_MULTI_LIST):
                pred_list.append(getattr(self, f"attr_multi_head_{id}")(features))
            for id, attr in enumerate(self.cfg.SECOND_ATTR_LIST):
                pred_list.append(getattr(self, f"second_attr_head_{id}")(features))
            for id, attr in enumerate(self.cfg.SECOND_ATTR_MULTI_LIST):
                pred_list.append(getattr(self, f"second_attr_multi_head_{id}")(features))
        elif self.cfg.ORGAN == 'breast':
            for id, attr in enumerate(self.cfg.ATTR_LIST):
                pred_list.append(getattr(self, f"attr_head_{id}")(features))
            # 增加了二级属性
            for id, attr in enumerate(self.cfg.SECOND_ATTR_MULTI_LIST):
                pred_list.append(getattr(self, f"attr_head_{id + len(self.cfg.ATTR_LIST)}")(features))

        return tuple(pred_list), features

    def _get_models_last(self, model):
        last_name = list(model._modules.keys())[-1]
        last_module = model._modules[last_name]
        return last_module, last_name
