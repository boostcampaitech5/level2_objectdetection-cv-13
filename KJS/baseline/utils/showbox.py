import os
import cv2
import numpy as np
from mmdet.apis import inference_detector 


def get_ann_box(image_id, coco_obj, root, mode): 
    '''image_id에 해당하는 이미지에 coco 객체로 부터 전달되는 annotation 정보를 이용해 box를 시각화 해주는 함수
    Args 
        image_id: int 
        coco_obj: COCO 
        root:     str
        mode: str ('train' or 'test') 
    Return 
        draw_img : ndarray
    '''
    labels_to_names_seq = ("General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    bbox_color = (0, 255, 0) 
    text_color = (255, 255, 255)
    
    ann_ids = coco_obj.getAnnIds(image_id)
    anns = coco_obj.loadAnns(ann_ids) 
    image_path = root + f'/{mode}/{str(image_id).zfill(4)}.jpg'
    draw_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  
    
    for ann in anns: 
        bbox = ann['bbox'] 
        
        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = int(bbox[0]) + int(bbox[2])
        y_max = int(bbox[1]) + int(bbox[3]) 

        caption = str(labels_to_names_seq[ann['category_id']])
        cv2.rectangle(draw_img, (x_min, y_min), (x_max, y_max), color=bbox_color, thickness=2)
        (w, h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, thickness=1)
        cv2.rectangle(draw_img, (x_min, y_min), (x_min+w+3, y_min+h+3), (0, 0, 0), -1) 
        cv2.putText(draw_img, caption, (x_min+2, y_min+16), cv2.FONT_HERSHEY_DUPLEX, color=text_color, fontScale=0.8, thickness=1) 

    return draw_img

def get_infer_box(model, image_id, root, score_thr=0.3): 
    ''' 모델로 부터 inference 된 box를 시각화 해주는 함수 
    Args 
        model:     model 객체 
        image_id:  int 
        root:      str
        score_thr: float
    Return 
        draw_img: ndarray
    '''
    labels_to_names_seq = ("General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing") 
    bbox_color = (0, 0, 225)
    text_color = (225, 225, 225)
    
    image_path = root + f'/train/{str(image_id).zfill(4)}.jpg'  
    
    draw_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) 
    
    results = inference_detector(model, draw_img)  
    
    for result_ind, result in enumerate(results): 
        if len(result) == 0: 
            continue 
        
        result_filtered = result[np.where(result[:, 4] >= score_thr)] 
        
        for i in range(len(result_filtered)): 
            
            x_min = int(result_filtered[i, 0])
            y_min = int(result_filtered[i, 1])
            x_max = int(result_filtered[i, 2])
            y_max = int(result_filtered[i, 3])
            
            caption = "{}: {:.2f}".format(labels_to_names_seq[result_ind], result_filtered[i, 4])
            cv2.rectangle(draw_img, (x_min, y_min), (x_max, y_max), color=bbox_color, thickness=2)
            (w, h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, thickness=1)
            cv2.rectangle(draw_img, (x_min, y_min), (x_min+w+3, y_min+h+3), (0, 0, 0), -1) 
            cv2.putText(draw_img, caption, (x_min+2, y_min+17), cv2.FONT_HERSHEY_DUPLEX, color=text_color, fontScale=0.8, thickness=1) 
            
    return draw_img  

if __name__=='__main__': 
    from pycocotools.coco import COCO 
    
    image_id = 0 
    coco = COCO('./pseudo_faster_rcnn_1024_adam.json') 
    root = '../dataset' 
    get_ann_box(image_id, coco, root, 'test')