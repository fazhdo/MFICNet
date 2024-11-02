import os
import random
import imgaug
import torch
import numpy as np
import random
from imgaug import augmenters as iaa
import cv2
from torch.utils.data import Dataset

from .builder import DATASETS

@DATASETS.register_module()
class SevenScenes_rgbd_aug():
    # root就是--data_path
    CLASSES = [
        'pumpkin'
    ]
    def __init__(self, root, dataset='7S', scene='heads', split='traiccn',
                    model='RGBDNET', aug='True', **kwargs):
        self.intrinsics_color = np.array([[525.0, 0.0,     320.0],
                       [0.0,     525.0, 240.0],
                       [0.0,     0.0,  1.0]])

        self.intrinsics_depth = np.array([[585.0, 0.0,     320.0],
                       [0.0,     585.0, 240.0],
                       [0.0,     0.0,  1.0]])

        self.intrinsics_depth_inv = np.linalg.inv(self.intrinsics_depth)
        self.intrinsics_color_inv = np.linalg.inv(self.intrinsics_color)
        self.model = model
        self.dataset = dataset
        self.aug = aug
        self.root = os.path.join(root, '7Scenes')
        self.calibration_extrinsics = np.loadtxt(os.path.join(self.root, 
                        'sensorTrans.txt'))
        self.scene = scene

        self.split = split                          # 模式选择
        self.obj_suffixes = ['.color.png','.pose.txt', '.depth.png',
                '.label.png']                       # 后缀
        self.obj_keys = ['color','pose', 'depth','label']
        self.n_sample_points = 4096 + 8192
        # 这里设定了训练/测试的图片
        with open(os.path.join(self.root, '{}{}'.format(self.split,         # ./data/7Scenes/tarin或test.txt
                '.txt')), 'r') as f:
            self.frames = f.readlines()
            if self.dataset == '7S' or self.split == 'test':
                # 列表['chess seq-03 frame-000000\n', 'chess seq-03 frame-000001\n', .............]
                self.frames = [frame for frame in self.frames \
                if self.scene in frame]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        # 遍历每一张图片
        frame = self.frames[index].rstrip('\n')         
        scene, seq_id, frame_id = frame.split(' ')      # chess seq-03 frame-000000
        objs = {}
        # /mnt/pipeline_2/datasets
        objs['color'] = '/mnt/pipeline_2/datasets/'+ self.scene + '/' + seq_id + '/' + frame_id + '.color.png'
        objs['pose'] = '/mnt/pipeline_2/datasets/' + self.scene + '/' + seq_id + '/' + frame_id + '.pose.txt'        # Twc
        objs['depth'] = '/mnt/pipeline_2/datasets/' + self.scene + '/' + seq_id + '/' + frame_id + '.depth.png'

        img = cv2.imread(objs['color'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pose = np.loadtxt(objs['pose'])                 # 位姿文件(np)

        depth = cv2.imread(objs['depth'],-1)

        depth[depth==65535] = 0
        depth = depth * 1.0

        # 返回深度图对齐到RGB图后，RGB相应的深度信息
        depth = get_depth(depth, self.calibration_extrinsics, self.intrinsics_color, self.intrinsics_depth_inv) # W H

        if self.split == 'test':
            cld, choose = dpt_2_cld(depth, 1, self.intrinsics_color)
            # cld1 = cld.data.cpu().numpy()
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
            #  print('cld after: ', np.info(cld)) # (12288, 3)
            #  print('choose after: ', np.info(choose)) # (1, 12288)
            #print('choose_2 after: ', np.info(choose_2)) # (12288,)

            img, cld, choose, pose = to_tensor_query(img, cld, choose, pose)
            ret = dict(img=img, cld=cld, choose=choose, pose=pose)
            return ret           # 返回torch类型的图片和位姿  


        
        dense_cld = dpt_2_dense_cld(depth, 1, self.intrinsics_color) # H W C 

        img, dense_cld, depth = data_aug(img, dense_cld, depth, self.aug)   # 进行数据增强

        """ from torchvision import transforms
        from torchvision.utils import save_image
        toPIL = transforms.ToPILImage() # 处理CHW格式图片
        pic = toPIL(img)
        pic.save('/home/dk/OFVL-VS2/debug/img3.png') """

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

        pose[0:3,3] = pose[0:3,3] * 1000
        gcld = get_coord(cld, pose) # [N 3]
        
        cld = point_data_aug(cld, self.aug)

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

        print(res_.shape)

        print(np.median(transl_err_list))
        print(np.median(rot_err_list))
        print(np.sum((res_[:, 0] <= 0.050) * (res_[:, 1] <= 5)) * 1. / len(res_))
        return dict(median_trans_error=np.median(res_[:, 0]),
                    median_rot_error=np.median(res_[:, 1]),
                    accuracy=np.sum((res_[:, 0] <= 0.050) * (res_[:, 1] <= 5)) * 1. / len(res_)
                    )
    
def dpt_2_dense_cld(dpt, cam_scale, K):
    high, width = dpt.shape
    
    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])

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

    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])

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
    
    # add noise
    

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
        ''' trans_x = random.randint(-3, 4)
        trans_y = random.randint(-3, 4)

        aug_affine = iaa.Affine(translate_px={"x": trans_x, "y": trans_y}) '''
        
        return img, dense_depth, depth

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


def data_aug_label(img, coord, mask, lbl, aug=True):
    img_h, img_w = img.shape[0:2]
    if aug:
        trans_x = random.uniform(-0.2, 0.2)
        trans_y = random.uniform(-0.2, 0.2)

        aug_add = iaa.Add(random.randint(-20, 20))

        scale = random.uniform(0.7, 1.5)
        rotate = random.uniform(-30, 30)
        shear = random.uniform(-10, 10)

        aug_affine = iaa.Affine(scale=scale, rotate=rotate,
                                shear=shear, translate_percent={"x": trans_x, "y": trans_y})
        aug_affine_lbl = iaa.Affine(scale=scale, rotate=rotate,
                                    shear=shear, translate_percent={"x": trans_x, "y": trans_y},
                                    order=0, cval=1)
        img = aug_add.augment_image(img)
    else:
        trans_x = random.randint(-3, 4)
        trans_y = random.randint(-3, 4)

        aug_affine = iaa.Affine(translate_px={"x": trans_x, "y": trans_y})
        aug_affine_lbl = iaa.Affine(translate_px={"x": trans_x, "y": trans_y},
                                    order=0, cval=1)

    padding = torch.randint(0, 255, size=(img_h,
                                          img_w, 3)).data.numpy().astype(np.uint8)
    padding_mask = np.ones((img_h, img_w)).astype(np.uint8)

    img = aug_affine.augment_image(img)
    coord = aug_affine.augment_image(coord)
    mask = aug_affine.augment_image(mask)
    mask = np.round(mask)
    lbl = aug_affine_lbl.augment_image(lbl)
    padding_mask = aug_affine.augment_image(padding_mask)
    img = img + (1 - np.expand_dims(padding_mask, axis=2)) * padding

    return img, coord, mask, lbl


def one_hot(x, N=25):
    one_hot = torch.FloatTensor(N, x.size(0), x.size(1)).zero_()
    one_hot = one_hot.scatter_(0, x.unsqueeze(0), 1)
    return one_hot


def to_tensor_label(img, coord_img, mask, lbl, N1=25, N2=25):
    img = img.transpose(2, 0, 1)
    coord_img = coord_img.transpose(2, 0, 1)

    img = img / 255.
    img = img * 2. - 1.

    coord_img = coord_img / 1000.

    img = torch.from_numpy(img).float()
    coord_img = torch.from_numpy(coord_img).float()
    mask = torch.from_numpy(mask).float()

    lbl = torch.from_numpy(lbl/1.0).long()
    lbl_oh = one_hot(lbl, N=N1)
    return img, coord_img, mask, lbl, lbl_oh


def to_tensor_query_label(img, pose):
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = img * 2. - 1.
    img = torch.from_numpy(img).float()
    pose = torch.from_numpy(pose).float()

    return img, pose

if __name__ == '__main__':
    root = '/home/dk/OFVL-VS2/data/'
    scene='pumpkin'
    split='train'
    dataset = SevenScenes_rgbd_aug(root = root, scene = scene, split = split)
    index = 1
    dataset.__getitem__(index)
    
    

