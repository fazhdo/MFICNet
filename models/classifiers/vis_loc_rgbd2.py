# Copyright (c) OpenMMLab. All rights reserved.
import torch
import time
from mmcv.runner import BaseModule

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck, build_loss
from ..heads import MultiLabelClsHead
from ..utils.augment import Augments
from .base import BaseClassifier
from ..losses import sevenScenesLoss
import sys
import cv2
import numpy as np


@CLASSIFIERS.register_module()
class RGBDNET(BaseClassifier):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None,
                 dataset="7Scenes"):
        super(RGBDNET, self).__init__(init_cfg)

        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
        self.loss = sevenScenesLoss.EuclideanLoss_with_Uncertainty()
        self.dataset = dataset

    def get_pose_err(self, pose_gt, pose_est):
            transl_err = np.linalg.norm(pose_gt[0:3, 3] - pose_est[0:3, 3])
            rot_err = pose_est[0:3, 0:3].T.dot(pose_gt[0:3, 0:3])
            rot_err = cv2.Rodrigues(rot_err)[0]  # 旋转向量 [3 1]
            rot_err = np.reshape(rot_err, (1, 3))  # 旋转向量 [1 3]
            rot_err = np.reshape(np.linalg.norm(rot_err, axis=1), -1) / np.pi * 180.  # 二范数即转角
            return transl_err, rot_err[0]

    def data_reconstruction(self, coord, pcoord):
        # coord = np.transpose(coord.cpu().data.numpy()[0, :, :, :], (1, 2, 0))  # [3 60 80]->[60 80 3]
        B, N, C = coord.shape
        coord = coord.cpu().data.numpy()[0, :, :]
        coord = np.ascontiguousarray(coord)
        pcoord = pcoord.cpu().data.numpy()[0, :, :]
        pcoord = np.ascontiguousarray(pcoord)
        coord = coord.reshape(-1, 3)
        pcoord = pcoord.reshape(-1, 2)

        coords_filtered = []
        coords_filtered_2D = []

        for i in range(N):
            coords_filtered.append(coord[i])
            coords_filtered_2D.append(pcoord[i])

        coords_filtered = np.vstack(coords_filtered)
        coords_filtered_2D = np.vstack(coords_filtered_2D)
        return coords_filtered, coords_filtered_2D

    def get_coord_from_choose(self, choose, img):
        high, width = img.shape[-2:] # 480 640
        coord_x = choose.to(torch.float32) / width
        coord_x = torch.floor(coord_x)
        coord_y = choose - coord_x * width
        coord_y = torch.floor(coord_y)
        coord_x = coord_x.to(torch.int32)
        coord_y = coord_y.to(torch.int32)
        coord = torch.concat((coord_y, coord_x), dim=1) # [4, 2, 12288]
        coord = coord.transpose(1, 2) # [4, 12288, 2]
        return coord

    def PNP_RANSAC(self, intrinsics_color, coords_filtered, coords_filtered_2D):
        sys.path.append("/home/4paradigm/XIETAO/MFICNet/mmclassification-v0.23.1/mmcls/models/classifiers/pnpransac")
        import pnpransac
        pose_solver = pnpransac.pnpransac(intrinsics_color[0, 0], intrinsics_color[1, 1], intrinsics_color[0, 2],
                                          intrinsics_color[1, 2])
        rot, transl = pose_solver.RANSAC_loop(coords_filtered_2D.astype(np.float64), coords_filtered.astype(np.float64),
                                              256)  # 预测结果,每次取256组点进行PNP Tcw
        return rot, transl

    def forward_test(self, img, cld, choose, pose, **kwargs):
        if self.dataset == "7Scenes":
            intrinsics_color = np.array([[525.0, 0.0,     320.0],
                           [0.0,     525.0, 240.0],
                           [0.0,     0.0,  1.0]])
        elif self.dataset == "12Scenes":
            intrinsics_color = np.array([[572.0, 0.0,     320.0],
                           [0.0,     572.0, 240.0],
                           [0.0,     0.0,  1.0]])
        elif self.dataset == "cambridge":
            intrinsics_color = np.array([[744.375, 0.0, 426.0],
                                        [0.0, 744.375, 240.0],
                                        [0.0, 0.0, 1.0]])
        elif self.dataset == "LTVL":
            intrinsics_color = np.array([[712.103, 0.0, 400.079],
                                          [0.0, 712.07, 299.316],
                                          [0.0, 0.0, 1.0]])
        

        pcoord = self.get_coord_from_choose(choose, img) # [4, 12288, 2]
        # Forward 
        data = [img, cld, choose]
        x = self.backbone(data)
        coord, uncer = self.head(x) # [4, 12288, 3], [4, 12288, 1]

        # coord2 = coord[0]
        # coord2 = coord2.data.cpu().numpy()
        # # np.savetxt('/home/dk/OFVL-VS2/debug/pred1.txt', coord2)
        # gt_lable2 = gt_lables[0]
        # gt_lable2 = gt_lable2.data.cpu().numpy()
        # # np.savetxt('/home/dk/OFVL-VS2/debug/label1.txt', gt_lable2)
        # pcoord2 = pcoord[0]
        # pcoord2 = pcoord2.data.cpu().numpy()
        # # np.savetxt('/home/dk/OFVL-VS2/debug/pred_pcoord.txt', pcoord2)

        # Coord data reconstruction
        coords_filtered, coords_filtered_2D = self.data_reconstruction(coord, pcoord)

        # Pnp ransac
        rot, transl = self.PNP_RANSAC(intrinsics_color, coords_filtered, coords_filtered_2D)

        # Get gt pose and est pose, and test error
        pose_gt = pose.cpu().numpy()[0, :, :]  # [4 4]
        pose_est = np.eye(4)  # [4 4]
        pose_est[0:3, 0:3] = cv2.Rodrigues(rot)[0].T  # Rwc
        pose_est[0:3, 3] = -np.dot(pose_est[0:3, 0:3], transl)  # twc
        transl_err, rot_err = self.get_pose_err(pose_gt, pose_est)
        return dict(trans_error_med=transl_err, rot_err_med=rot_err)

    def forward_train(self, img, cld, choose, gt_lables, **kwargs):
        # Forward 
        data = [img, cld, choose] # [B, C, H, W]
        
        x = self.backbone(data)

        coord, uncer = self.head(x) # [4, 12288, 3] [4, 12288, 1]
        # Compute loss
        loss, accuracy = self.loss(coord, gt_lables, uncer)
        """ coord2 = coord[0]
        coord2 = coord2.data.cpu().numpy()
        np.savetxt('/home/dk/OFVL-VS2/debug/pred.txt', coord2)
        gt_lable2 = gt_lables[0]
        gt_lable2 = gt_lable2.data.cpu().numpy()
        np.savetxt('/home/dk/OFVL-VS2/debug/label.txt', gt_lable2)
        exit(-1) """
        losses = dict(loss=loss, accuracy=accuracy)
        return losses

    def forward(self, img, cld, choose, gt_lables=None, pose=None, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_meta are single-nested (i.e. Tensor and
        List[dict]), and when `resturn_loss=False`, img and img_meta should be
        double nested (i.e.  List[Tensor], List[List[dict]]), with the outer
        list indicating test time augmentations.
        """
        if return_loss:
            assert choose is not None
            return self.forward_train(img, cld, choose, gt_lables, **kwargs)
        else:
            return self.forward_test(img, cld, choose, pose, **kwargs)
    
    def extract_feat(self, img, stage='neck'):
        pass

    def simple_test(self, img, img_metas=None, **kwargs):
        pass