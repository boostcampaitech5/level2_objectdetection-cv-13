from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
exp_num = "exp19"
cfg = Config.fromfile(f'./configs/_custom/{exp_num}/htc_swinL_fpn_1x_coco_pseudo.py')

cfg.log_config.hooks = [dict(type='TextLoggerHook'),
                            dict(type='MMDetWandbHook',
                                init_kwargs=dict(   project = 'Object-Detection_jh',
                                                    entity = 'connect-cv-13_2',
                                                    name = f"{exp_num}_htc_swinL_fpn_1x_coco_pseudo.jh"),
                                interval=500,
                                log_checkpoint=False,
                                log_checkpoint_metadata=False,
                                num_eval_images=100,
                                bbox_score_thr=0.3,
                                )]
root='../../dataset/'

# dataset config 수정

cfg.seed = 2022
cfg.gpu_ids = [0]
cfg.work_dir = f'./work_dirs/{exp_num}'

cfg.checkpoint_config = dict(max_keep_ckpts=2, interval=1)
cfg.evaluation = dict(save_best='bbox_mAP_50',metric='bbox')
cfg.device = get_device()

datasets = [build_dataset(cfg.data.train)]

model = build_detector(cfg.model)
model.init_weights()

train_detector(model, datasets[0], cfg, distributed=False, validate=True)