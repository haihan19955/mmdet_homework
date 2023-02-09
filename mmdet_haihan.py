_base_ = './mask_rcnn_r101_caffe_fpn_1x_coco.py'
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained')),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))
dataset_type = 'COCODataset' 
classes = ('balloon',) 
data = dict(
    train=dict(
        img_prefix='/home/haihan/projects/mmlab/data/balloon/train/',
        classes=classes,
        ann_file='/home/haihan/projects/mmlab/data/balloon/train/annotation_coco.json'), 
    val=dict(
        img_prefix='/home/haihan/projects/mmlab/data/balloon/val/',  
        classes=classes,
        ann_file='/home/haihan/projects/mmlab/data/balloon/val/annotation_coco.json'),
    test=dict(
        img_prefix='/home/haihan/projects/mmlab/data/balloon/val/',  
        classes=classes,
        ann_file='/home/haihan/projects/mmlab/data/balloon/val/annotation_coco.json')) 
 
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

load_from = '/home/haihan/projects/mmlab/checkpoints/mask_rcnn_r101_caffe_fpn_1x_coco_20200601_095758-805e06c1.pth'
work_dir = './work_dirs/mask_rcnn_haihan'