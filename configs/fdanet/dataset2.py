_base_ = [
    '../_base_/default_runtime.py'
]

# SevenScenes TWESCENES livl_rgbd SevenScenes_rgbd_aug LTVL_rgbd
dataset_type = 'mmcls.LTVL_rgbd_offset'
scene='dataset2-s'
train_pipeline = []
test_pipeline = []
eval_pipeline = []
custom_imports = dict(imports=['mmcls.models'])
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.0,
    warmup='linear',
    warmup_iters=400,
    warmup_ratio=1e-4)
# optimizer config
optimizer = dict(type='AdamW', lr=0.0002 * 10, weight_decay=0.1)
optimizer_config = dict(grad_clip=dict(max_norm=100.0))
# checkpoint saving
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(interval=5000, max_keep_ckpts=6)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook')
    ])

# point net meta
# num_points=(2048, 1024, 512, 256)
# radius=(0.2, 0.4, 0.8, 1.2)
# num_samples=(64, 32, 16, 16)
# sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),(128, 128, 256))
# fp_channels=((256, 256), (256, 256), (256, 256), (256, 256))

model = dict(
    type='mmcls.RGBDNET',
    dataset='LTVL',
    backbone=dict(
        type='SEResNet_PointNet2',
        depth=18,
        stem_channels=16,
        expansion = 1,
        strides=(1, 1, 2, 2),
        use_maxpool=False,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        Pn_in_channels = 3,
        Pn_num_points=(2048, 1024, 512, 256),
        Pn_radius=(0.05, 0.1, 0.2, 0.4),
        Pn_num_samples=(64, 32, 16, 16),
        Pn_sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),(128, 128, 256)),
        Pn_fp_channels=((256, 256), (256, 256), (256, 256), (256, 128)),
        psp_param = dict(in_channels=512, channels=512, num_classes=19)
        ),
    head=dict(
        type='RGBDHead',
        in_channel=1792))

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        scene=scene,
        split='train'),
    val=dict(
        type=dataset_type,
        scene=scene,
        split='test'),
    test=dict(
        type=dataset_type,
        scene=scene,
        split='test'))

evaluation = dict(interval=5000, save_best='median_trans_error', by_epoch=False, rule='less')
find_unused_parameters = True
# resume_from = "/mnt/pipeline_2/work_dirs/CEILING_cat/iter_30000.pth"