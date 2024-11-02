import os
import random

import cv2
import imgaug
import numpy as np
import torch
from imgaug import augmenters as iaa
from torch.utils.data import Dataset

from .builder import DATASETS


@DATASETS.register_module()
class LTVL_rgbd_offset(Dataset):
    CLASSES = [
        'chess'
    ]

    def __init__(self, split, scene='CEILING', aug=True, **kwargs):
        self.intrinsics_color = np.array([[712.103, 0.0, 400.079],
                                          [0.0, 712.07, 299.316],
                                          [0.0, 0.0, 1.0]])
        self.intrinsics_color_inv = np.linalg.inv(self.intrinsics_color)  # 颜色相机内参的逆
        self.split = split
        self.scene = scene
        self.data_rgb = 'train_list_rgb.txt' if self.split == 'train' else 'test_list_rgb.txt'
        self.data_depth = 'train_list_depth.txt' if self.split == 'train' else 'test_list_depth.txt'
        self.aug = aug
        self.n_sample_points = 7500
        self.frame_rgb = []
        self.frame_depth = []
        self.RT = []
                
        with open('/mnt/pipeline_2/datasets/LTVL/' + self.scene + '/' + self.data_rgb) as f:
            self.frames_rgb = f.readlines()
            for j in self.frames_rgb:
                self.frame_rgb.append((j))

        with open('/mnt/pipeline_2/datasets/LTVL/' + self.scene + '/' + self.data_depth) as f:
            self.frames_depth = f.readlines()
            for j in self.frames_depth:
                self.frame_depth.append((j))

        # sparse_index = open("/mnt/share/sda-8T/dk/Laser/sparse_index/Park1/mohu.txt", "rb").readlines()
        # for i, n in enumerate(sparse_index):
        #     sparse_index[i] = n.strip().decode("utf-8")
        # self.frame = sparse_index

    def __len__(self):
        return len(self.frame_rgb)

    def __getitem__(self, index):
        id_rgb = self.frame_rgb[index][0:4]
        id_depth = self.frame_depth[index][0:4]
        objs = {}

        # seq_id = "no4"
        # id = "514"

        objs[
            'color'] = '/mnt/pipeline_2/datasets/LTVL/' + self.scene + '/color' + '/' + id_rgb + '.png'
        objs[
            'depth'] = '/mnt/pipeline_2/datasets/LTVL/' + self.scene + '/depth' + '/' + id_depth + '.png'  # Twc
        objs[
            'pose'] = '/mnt/pipeline_2/datasets/LTVL/' + self.scene + '/pose' + '/' + id_depth + '.txt'

        try:
            img = cv2.imread(objs['color'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pose = np.loadtxt(objs['pose'])                 # 位姿文件(np)
            depth = cv2.imread(objs['depth'],-1)
        except:
            print('load data error')
            exit()

        if self.split == 'test':
            cld, choose = dpt_2_cld(depth, 5, self.intrinsics_color)

            choose = np.array([choose])

            choose_2 = np.array([i for i in range(len(choose[0, :]))])
            if len(choose_2) > self.n_sample_points:
                c_mask = np.zeros(len(choose_2), dtype=int)
                c_mask[:self.n_sample_points] = 1
                np.random.shuffle(c_mask)
                choose_2 = choose_2[c_mask.nonzero()]
            else:
                if len(choose_2) == 0:
                    choose_2 = np.zeros(1)
                    choose_2 = np.pad(choose_2, (0, self.n_sample_points-len(choose_2)), 'wrap')
                    cld = np.zeros((self.n_sample_points, 3))
                    choose = np.zeros((1, self.n_sample_points))
                else:
                    choose_2 = np.pad(choose_2, (0, self.n_sample_points-len(choose_2)), 'wrap')
            # print('cld before: ', np.info(cld)) # (196791, 3)
            # print('choose before: ', np.info(choose)) # (1, 196791)
            choose_2 = choose_2.astype(np.int64)
            cld = cld[choose_2, :]
            choose = choose[:, choose_2]
            pose[0:3,3] = pose[0:3,3] / 1000

            img, cld, choose, pose = to_tensor_query(img, cld, choose, pose)
            ret = dict(img=img, cld=cld, choose=choose, pose=pose)
            return ret

        dense_cld = dpt_2_dense_cld(depth, 5, self.intrinsics_color) # H W C 

        img, dense_cld, depth = data_aug(img, dense_cld, depth, self.aug)   # 进行数据增强

        cld, choose = dense_cld_2_cld(dense_cld, depth)

        choose = np.array([choose])

        choose_2 = np.array([i for i in range(len(choose[0, :]))])

        if len(choose_2) > self.n_sample_points:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:self.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            if len(choose_2) == 0:
                choose_2 = np.zeros(1)
                choose_2 = np.pad(choose_2, (0, self.n_sample_points-len(choose_2)), 'wrap')
                cld = np.zeros((self.n_sample_points, 3))
                choose = np.zeros((1, self.n_sample_points))
            else:
                choose_2 = np.pad(choose_2, (0, self.n_sample_points-len(choose_2)), 'wrap')

        # print('cld before: ', np.info(cld)) # (196791, 3)
        # print('choose before: ', np.info(choose)) # (1, 196791)
        choose_2 = choose_2.astype(np.int64)
        cld = cld[choose_2, :]
        choose = choose[:, choose_2]

        """ np.savetxt('/home/dk/OFVL-VS2/debug/cld2.txt', cld)
        print(pose)
        exit() """

        pose[0:3,3] = pose[0:3,3]
        gcld = get_coord(cld, pose) # [N 3]

        # cld = point_data_aug(cld, self.aug)

        img, cld, choose, gcld = to_tensor(img, cld, choose, gcld)

        # img :     [3, 480, 640] 
        # cld :     [12288, 3] 
        # choose :  [1, 12288] 
        # gcld :    [12288, 3] 
        ret = dict(img=img, cld=cld, choose=choose, gt_lables=gcld)
        return ret

    def evaluate(self, results, *args, **kwargs):
        transl_err_list = list()
        rot_err_list = list()
        for i in range(len(results)):
            transl_err_list.append(results[i]['trans_error_med'])
            rot_err_list.append(results[i]['rot_err_med'])
        res_ = np.array([transl_err_list, rot_err_list]).T

        print(np.median(transl_err_list))
        print(np.median(rot_err_list))
        print(np.sum((res_[:, 0] <= 0.050) * (res_[:, 1] <= 5)) * 1. / len(res_))
        return dict(median_trans_error=np.median(res_[:, 0]),
                    median_rot_error=np.median(res_[:, 1]),
                    accuracy=np.sum((res_[:, 0] <= 0.050) * (res_[:, 1] <= 5)) * 1. / len(res_)
                    )


   
def dpt_2_dense_cld(dpt, cam_scale, K):
    high, width = dpt.shape
    
    xmap = np.array([[j for i in range(width)] for j in range(high)])
    ymap = np.array([[i for i in range(width)] for j in range(high)])

    dpt_flatten = dpt.flatten()[:, np.newaxis].astype(np.float32)  # 480 * 640  1
    xmap_flatten = xmap.flatten()[:, np.newaxis].astype(np.float32) # 480 * 640  1
    ymap_flatten = ymap.flatten()[:, np.newaxis].astype(np.float32) # 480 * 640  1

    pt2 = dpt_flatten / cam_scale
    cam_cx, cam_cy = K[0][2], K[1][2]
    cam_fx, cam_fy = K[0][0], K[1][1]
    pt0 = (ymap_flatten - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_flatten - cam_cy) * pt2 / cam_fy
    cld = np.concatenate((pt0, pt1, pt2), axis=1) # N  3
    dense_cld = cld.reshape(high, width, 3) # H W 3 

    """ print(xmap_mskd)

    print('choose after: ', np.info(choose)) #  (204930,)
    print('xmap_mskd after: ', np.info(xmap_mskd)) # (204930, 1)
    print('xmap after: ', np.info(xmap)) # (480, 640) """
    return dense_cld

def dense_cld_2_cld(dense_cld, depth):
    msk_dp = np.ones_like(depth) 
    msk_dp[depth == 0] = 0
    choose = msk_dp.flatten().nonzero()[0].astype(np.int32)
    """ if len(choose) < 12288:
        print("unpredicte condition!!!")
        return None, choose """

    cld = np.reshape(dense_cld, (-1, 3)) # high*width 3
    cld = cld[choose, :] # N 3
    return cld, choose

def dpt_2_cld(dpt, cam_scale, K):
    msk_dp = np.ones_like(dpt)
    msk_dp[dpt == 0] = 0
    choose = msk_dp.flatten().nonzero()[0].astype(np.int32)
    """ if len(choose) < 12288:
        print("unpredicte condition!!!")
        return None, choose """

    xmap = np.array([[j for i in range(800)] for j in range(600)])
    ymap = np.array([[i for i in range(800)] for j in range(600)])

    dpt_mskd = dpt.flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_mskd = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_mskd = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

    pt2 = dpt_mskd / cam_scale
    cam_cx, cam_cy = K[0][2], K[1][2]
    cam_fx, cam_fy = K[0][0], K[1][1]
    pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
    cld = np.concatenate((pt0, pt1, pt2), axis=1)

    """ print(xmap_mskd)

    print('choose after: ', np.info(choose)) #  (204930,)
    print('xmap_mskd after: ', np.info(xmap_mskd)) # (204930, 1)
    print('xmap after: ', np.info(xmap)) # (480, 640) """
    return cld, choose

# depth：深度图[480 640]
# 返回深度图对齐到RGB图后，RGB相应的深度信息
def get_depth(depth, calibration_extrinsics, intrinsics_color,
              intrinsics_depth_inv):
    """Return the calibrated depth image (7-Scenes).
    Calibration parameters from DSAC (https://github.com/cvlab-dresden/DSAC)
    are used.
    """
    '''
    利用深度摄像头内参矩阵把深度平面坐标（深度图坐标）转换到深度摄像头空间坐标，
    再利用外参计算旋转矩阵和平移矩阵，把深度摄像头空间坐标转换到RGB摄像头空间坐标，
    最后利用RGB摄像头内参矩阵把RGB摄像头空间坐标转换到RGB平面坐标（RGB图坐标）。
    这里只记录一下最终测试程序的思路：
    '''
    img_height, img_width = depth.shape[0], depth.shape[1]
    depth_ = np.zeros_like(depth)  # [480 640]
    x = np.linspace(0, img_width - 1, img_width)  # 640
    y = np.linspace(0, img_height - 1, img_height)  # 480

    xx, yy = np.meshgrid(x, y)  # 坐标网格化[img_width img_height]
    xx = np.reshape(xx, (1, -1))  # [1, img_width*img_height]
    yy = np.reshape(yy, (1, -1))  # [1, img_width*img_height]
    ones = np.ones_like(xx)  # [1, img_width*img_height]

    pcoord_depth = np.concatenate((xx, yy, ones), axis=0)  # [3, img_width*img_height], 像素坐标
    depth = np.reshape(depth, (1, img_height * img_width))  # [1, img_width*img_height]

    ccoord_depth = np.dot(intrinsics_depth_inv, pcoord_depth) * depth  # 像素坐标-->归一化坐标-->相机坐标[3, img_width*img_height]

    ccoord_depth[1, :] = - ccoord_depth[1, :]
    ccoord_depth[2, :] = - ccoord_depth[2, :]

    ccoord_depth = np.concatenate((ccoord_depth, ones), axis=0)  # [4, img_width*img_height]
    ccoord_color = np.dot(calibration_extrinsics, ccoord_depth)  # [3, img_width*img_height],RGB相机坐标

    ccoord_color = ccoord_color[0:3, :]
    ccoord_color[1, :] = - ccoord_color[1, :]
    ccoord_color[2, :] = depth

    pcoord_color = np.dot(intrinsics_color, ccoord_color)  # RGB像素坐标*Z
    pcoord_color = pcoord_color[:, pcoord_color[2, :] != 0]

    pcoord_color[0, :] = pcoord_color[0, :] / pcoord_color[2, :] + 0.5  # RGB像素坐标
    pcoord_color[0, :] = pcoord_color[0, :].astype(int)
    pcoord_color[1, :] = pcoord_color[1, :] / pcoord_color[2, :] + 0.5
    pcoord_color[1, :] = pcoord_color[1, :].astype(int)
    pcoord_color = pcoord_color[:, pcoord_color[0, :] >= 0]
    pcoord_color = pcoord_color[:, pcoord_color[1, :] >= 0]

    pcoord_color = pcoord_color[:, pcoord_color[0, :] < img_width]
    pcoord_color = pcoord_color[:, pcoord_color[1, :] < img_height]

    depth_[pcoord_color[1, :].astype(int),
           pcoord_color[0, :].astype(int)] = pcoord_color[2, :]

    return depth_


def get_coord(cld, pose):
    """Generate the ground truth scene coordinates from depth and pose.
    """
    ones = np.ones_like(cld[:, 0]) # [N 1]
    ones = ones[:, np.newaxis]

    ccoord = np.concatenate((cld, ones), axis=1)  # 相机坐标 [N 4]
    ccoord = ccoord.transpose(1, 0) # [4, N]
    scoord = np.dot(pose, ccoord)  # 世界坐标 [4 640*480]
    scoord = scoord.transpose(1, 0) # [N, 4]
    scoord = scoord[:, 0:3] # [N, 3]
    return scoord



# 点云数据增强
def point_data_aug(batch_data, aug=True):
    N, C = batch_data.shape[0:2]
    rand_data = random.randint(1, 10)
    if aug and rand_data < 3:
        # $->>>> first stage: rotation <<<<-$ #
        angle_sigma=0.06
        angle_clip=0.18
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        # 对xyz三个轴方向随机生成一个旋转角度
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        # 根据公式构建三个轴方向的旋转矩阵
        Rx = np.array([[1,0,0],
                    [0,np.cos(angles[0]),-np.sin(angles[0])],
                    [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                    [0,1,0],
                    [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                    [np.sin(angles[2]),np.cos(angles[2]),0],
                    [0,0,1]])
        # 按照内旋方式:Z-Y-X旋转顺序获得整体的旋转矩阵
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[:,0:3]
        # 分别对坐标与法向量进行旋转,整体公式应该为: Pt = (Rz * Ry * Rx) * P
        rotated_data[:,0:3] = np.dot(shape_pc.reshape((-1, 3)), R)

        # $->>>> second stage: noise <<<<-$ #
        sigma=0.005 * 1000
        clip=0.03 * 1000
        assert(clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
        jittered_data += rotated_data    # 添加噪声

        # $->>>> third stage: transformation <<<<-$ #
        shift_range=0.1 * 1000
        shifts = np.random.uniform(-shift_range, shift_range, (3))    # 对每个batch的点云设置一个随机的移动偏差
        jittered_data[:,:] += shifts[:]    # 每个点都进行移动
        
        # $->>>> fourth stage: scale <<<<-$ #
        scale_low=0.9
        scale_high=1.1
        scales = np.random.uniform(scale_low, scale_high)    # 0.8~1.25间的随机缩放
        jittered_data[:,:] *= scales  # 每个点都进行缩放

        return jittered_data
    
    return batch_data


# 数据增强操作
def data_aug(img, dense_depth, depth, aug=True, sp_coords=None):
    img_h, img_w = img.shape[0:2]
    rand_data = random.randint(1, 10)
    if aug:
        trans_x = random.uniform(-0.2, 0.2)  # 平移
        trans_y = random.uniform(-0.2, 0.2)

        aug_add = iaa.Add(random.randint(-20, 20))

        scale = random.uniform(0.7, 1.5)  # 缩放
        rotate = random.uniform(-30, 30)  # 旋转
        shear = random.uniform(-10, 10)  # 裁剪

        aug_affine = iaa.Affine(scale=scale, rotate=rotate,
                                shear=shear, translate_percent={"x": trans_x, "y": trans_y})
        aug_affine_lbl = iaa.Affine(scale=scale,rotate=rotate,
                    shear=shear,translate_percent={"x": trans_x, "y": trans_y},
                    order=0,cval=1)
        img = aug_add.augment_image(img)
    else:
        trans_x = random.randint(-3, 4)
        trans_y = random.randint(-3, 4)

        aug_affine = iaa.Affine(translate_px={"x": trans_x, "y": trans_y})

    padding = torch.randint(0, 255, size=(img_h, img_w, 3)).data.numpy().astype(np.uint8)
    padding_mask = np.ones((img_h, img_w)).astype(np.uint8)

    img = aug_affine.augment_image(img)
    depth = aug_affine.augment_image(depth)
    dense_depth = aug_affine.augment_image(dense_depth)
    """ from torchvision import transforms
    from torchvision.utils import save_image
    toPIL = transforms.ToPILImage() # 处理CHW格式图片
    pic = toPIL(img)
    pic.save('/home/dk/OFVL-VS2/debug/img3.png')
    import pdb;pdb.set_trace()
    pic = toPIL(depth.astype(np.int32))
    pic.save('/home/dk/OFVL-VS2/debug/depth3.png') """
    padding_mask = aug_affine.augment_image(padding_mask)
    img = img + (1 - np.expand_dims(padding_mask, axis=2)) * padding
    return img, dense_depth, depth


# img [480 640 3]
# coord_img [60, 80, 3]
# mask [60 80]
# cld [N 3]
# choose [N 1]
# gcld [N 3]
def to_tensor(img, cld, choose, gcld):
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = img * 2. - 1.
    img = torch.from_numpy(img).float()

    cld = cld / 1000.
    cld = torch.from_numpy(cld).float()

    choose = torch.from_numpy(choose.astype(np.int64))

    gcld = gcld / 1000.
    gcld = torch.from_numpy(gcld).float()
    return img, cld, choose, gcld


def to_tensor_query(img, cld, choose, pose):
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = img * 2. - 1.
    img = torch.from_numpy(img).float()
    pose = torch.from_numpy(pose).float()
    cld = torch.from_numpy(cld).float()
    cld = cld / 1000
    choose = torch.from_numpy(choose.astype(np.int64))
    return img, cld, choose, pose
