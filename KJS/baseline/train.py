from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device 

import wandb 
import plotly 


IMG_ROOT = '/opt/ml/level2_project/dataset' 
# IMG_ROOT = '/opt/ml/level2_project/pseudo_dataset'
CLASSES = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing") 


##################### 수정 ############################ 
# 세부 수정은 실험별 config 파일에서 직접 하기 
wandb_exp_name = 'exp15_retinanet_finetune'
config_file_path = '/opt/ml/level2_project/baseline/configs/retinanet/retinanet_r101_fpn_1x_coco.py'
checkpoints_path = '/opt/ml/level2_project/baseline/checkpoints/exp14_retinanet_epoch_11.pth' 
# checkpoints_path = '/opt/ml/level2_project/submission/exp5/epoch_12.pth'
fold = 2
work_dirs_path = '/opt/ml/level2_project/baseline/work_dirs/exp15'
epochs = 15
RESIZE = (1024, 1024)
#######################################################

def set_config(cfg): 
    
    cfg.dataset_type = "CocoDataset" 

    cfg.data.train.classes = CLASSES
    # cfg.data.train.ann_file = IMG_ROOT + f'/pseudo_filtered_groupskfold/train_fold{fold}.json' # train json 정보
    cfg.data.train.ann_file = IMG_ROOT + '/groupskfold/train_fold3.json'
    cfg.data.train.img_prefix = IMG_ROOT
    
    cfg.data.val.classes = CLASSES
    # cfg.data.val.ann_file = IMG_ROOT + f'/pseudo_filtered_groupskfold/valid_fold{fold}.json'
    cfg.data.val.ann_file = IMG_ROOT + '/groupskfold/valid_fold3.json'
    cfg.data.val.img_prefix = IMG_ROOT

    cfg.data.test.classes = CLASSES 
    cfg.data.test.ann_file = IMG_ROOT + '/test.json'  
    cfg.data.test.img_prefix = IMG_ROOT 
    
    cfg.data.train.pipeline[2]['img_scale'] = RESIZE # Resize
    cfg.data.val.pipeline[1]['img_scale'] = RESIZE # Resize
    cfg.data.test.pipeline[1]['img_scale'] = RESIZE # Resize
    cfg.data.train.pipeline[2]['img_scale'] = RESIZE # Resize
    cfg.train_pipeline[2]['img_scale'] = RESIZE 
    cfg.test_pipeline[1]['img_scale'] = RESIZE 

    cfg.load_from = checkpoints_path 
    
    # cfg.model.roi_head.bbox_head.num_classes = 10 
    cfg.model.bbox_head.num_classes=10  # retinanet에서 사용 

    cfg.runner.max_epochs = epochs
    cfg.optimizer=dict(type='Adam', lr=0.000025, weight_decay=0.0001)
    cfg.lr_config = dict(
        policy='CosineAnnealing', 
        warmup='linear', 
        warmup_iters=500, 
        warmup_ratio=0.001, 
        min_lr_ratio=0.001
    )

    cfg.data.samples_per_gpu = 4
    cfg.data.workers_per_gpu = 4
    cfg.seed = 2022
    cfg.gpu_ids = [0]
    cfg.work_dir = work_dirs_path
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=5, interval=1)
    cfg.device = get_device() 

    cfg.evaluation.metric = ["bbox"]


    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
            init_kwargs={
            'project': 'ObjectDetection-MMDetection' , 
            'entity': 'connect-cv-13_2',
            "tags" : ["practice_wandb_first"], 
            'name' : wandb_exp_name  
            },
            interval=50,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=30,
            bbox_score_thr=0.3),
        ] 
    # cfg.evaluation.interval = 1

def train(cfg):
    
    datasets = [build_dataset(cfg.data.train)]
    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    # model.init_weights() 
    train_detector(model, datasets[0], cfg, distributed=False, validate=True) 
    
    
if __name__ == "__main__":
    cfg = Config.fromfile(config_file_path) 
    
    # cfg 설정
    set_config(cfg)
    
    # cfg 확인 
    
    # print(cfg.pretty_text)
    
    # 학습 
    train(cfg)