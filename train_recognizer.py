from mmcv import Config
from mmcv.runner import set_random_seed

import os.path as osp

from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.apis import train_model

import mmcv

root = "/media/users/nunziati/adip/"

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# CHANGE AT EACH EXPERIMENT

# IRCSN:
cfg = Config.fromfile(f'{root}mmaction2/configs/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py')

# TPN:
# cfg = Config.fromfile('./mmaction2/configs/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb.py')

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Modify dataset type and path
cfg.dataset_type = 'VideoDataset'
cfg.data_root = f'{root}train/'
cfg.data_root_val = f'{root}validation/'
cfg.data_root_test = f'{root}test/'
cfg.ann_file_train = f'{root}training.csv'
cfg.ann_file_val = f'{root}validation.csv'
cfg.ann_file_test = f'{root}test.csv'

cfg.data.test.type = 'VideoDataset'
cfg.data.test.ann_file = f'{root}test.csv'
cfg.data.test.data_prefix = f'{root}test/'

cfg.data.train.type = 'VideoDataset'
cfg.data.train.ann_file = f'{root}training.csv'
cfg.data.train.data_prefix = f'{root}train/'

cfg.data.val.type = 'VideoDataset'
cfg.data.val.ann_file = f'{root}validation.csv'
cfg.data.val.data_prefix = f'{root}validation/'

cfg.train_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='ColorJitter'),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
cfg.val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='ColorJitter'),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
cfg.test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])]

cfg.data = dict(
    videos_per_gpu=2,
    workers_per_gpu=1,
    test_dataloader=dict(videos_per_gpu=1),
    val_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=cfg.dataset_type,
        ann_file=cfg.ann_file_train,
        data_prefix=cfg.data_root,
        pipeline=cfg.train_pipeline),
    val=dict(
        type=cfg.dataset_type,
        ann_file=cfg.ann_file_val,
        data_prefix=cfg.data_root_val,
        pipeline=cfg.val_pipeline),
    test=dict(
        type=cfg.dataset_type,
        ann_file=cfg.ann_file_test,
        data_prefix=cfg.data_root_test,
        pipeline=cfg.test_pipeline))


# The flag is used to determine whether it is omnisource training
cfg.setdefault('omnisource', False)
# Modify num classes of the model in cls_head
cfg.model.cls_head.num_classes = 14


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# We can use the pre-trained TSN model. CHANGE AT EACH EXPERIMENT

# IRCSN:
cfg.load_from = 'https://download.openmmlab.com/mmaction/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_20200812-9037a758.pth'

# TPN:
# cfg.load_from = 'https://download.openmmlab.com/mmaction/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb-44362b55.pth'

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Set up working dir to save files and logs. CHANGE AT EACH EXPERIMENT

# Use the name of the experiment, find it in the google drive sheet
cfg.work_dir = f'{root}experiments/exp02_ircsn_sgd_unbal'

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Training hyperparameters. CHANGE AT EACH EXPERIMENT

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = cfg.optimizer.lr / 8
cfg.total_epochs = 50
cfg.optimizer.type='SGD' # OPTIONS: 'SGD', 'Adam'
cfg.optimizer.momentum=0.9 # ONLY FOR SGD
cfg.optimizer.weight_decay=0.0001

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

if cfg.optimizer.type == 'Adam': del cfg.optimizer.momentum



# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 5
# We can set the log print interval to reduce the the times of printing log
cfg.log_config.interval = 5

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# Save the best
cfg.evaluation.save_best='auto'
cfg.lr_config = dict(policy='step', step=[75, 125])



# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')


# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the recognizer
model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_model(model, datasets, cfg, distributed=False, validate=True)