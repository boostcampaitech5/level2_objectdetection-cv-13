from tqdm import tqdm
import os
import json
from collections import defaultdict
import random


# split data
def split_data(train_ratio:float=0.8, exclude:bool=False, excluded_id:list=[], file_path:str='train.json', licence:bool=False) -> list:
    """
    train_ratio : ratio of training data to base data
    excluded_id : the list of ids of the image that miss labeled or excluded for some 
    exclude : whether or not to exclude miss labeled data
    file_path : path to the file to load
    licence : whether or not to include a license
    return : list of train data and validation data
    """
    
    
    # load json file and exclude data
    def load_json(file_path=file_path, excluded_id=excluded_id, exclude=exclude):
        with open (file_path, 'r') as base_data:
            base_data_json = json.load(base_data)
        print(f'| Load [{file_path}] complete |')
        print('')
    
        if exclude:
            print(f'| Excluding start |')
            idx = 0
            progress = 1
            
            while True:
                image_length = len(base_data_json['images'])
                  
                if idx > image_length-1:
                    break
                if base_data_json['images'][idx]['id'] in excluded_id:
                    del base_data_json['images'][idx]
                else:
                    idx += 1
            idx = 0
            progress = 1   
            
            while True:
                annot_length = len(base_data_json['annotations'])
                
                if idx > annot_length-1:
                    break
                if base_data_json['annotations'][idx]['image_id'] in excluded_id:
                    del base_data_json['annotations'][idx]
                else:
                    idx += 1 
            print(f'| Excluding end |')
            print('')
            
        return base_data_json
    
              
    # sorting data with number of categories
    def sorting_data():
        categories = []
        for category in base_data_json['categories']:
            categories.append(category['id'])

        annotations = base_data_json['annotations']

        data_category = defaultdict(lambda: [0])

        for anot in annotations:
            category = anot['category_id']
            data_category[category][0] += 1

        data_category_list = sorted([[key, value[0]] for key, value in data_category.items()], key=lambda x:x[1])

        return data_category_list


    # classify image with categories
    def classify_image():
        inner_basic_dict = {
            0:0,
            1:0,
            2:0,
            3:0,
            4:0,
            5:0,
            6:0,
            7:0,
            8:0,
            9:0,
        }

        num_dict = defaultdict(lambda: inner_basic_dict.copy())
        class_dict = defaultdict(lambda: [])
        
        print(f'| image classify start |')
        
        pbar = tqdm(base_data_json['images'])
        
        for image in pbar:
            for anot in base_data_json['annotations']:
                if anot['image_id'] == image['id']:
                    num_dict[image['id']][anot['category_id']] += 1

                    if image['id'] not in class_dict[anot['category_id']]:
                        class_dict[anot['category_id']].append(image['id'])
        pbar.close()
        
        print(f'| image classify end |')
        print('')

        return num_dict, class_dict
    

    # init trainset and validset
    def init_dataset(licence=licence):
        
        
        train_set = {
            'images':[],
            'annotations':[]
        }
        valid_set = {
            'images':[],
            'annotations':[]
        }

        category_dict = {
            'categories': base_data_json['categories']
        }
        
        train_set.update(category_dict)
        valid_set.update(category_dict)
        if licence:
            licence_dict = {
                'info':base_data_json['info'],
                'licenses':base_data_json['licenses'],
            }

            train_set.update(licence_dict)
            valid_set.update(licence_dict)
            
        return train_set, valid_set


    # split dataset locally
    def split_data_local(class_data, classified_data, class_info, train_ratio):
        train_list = []
        valid_list = []
        valid_sum = 0

        class_data_local = class_data[class_info[0]].copy()
        random.shuffle(class_data_local)
        
        data_length = len(class_data_local)

        with tqdm(total=data_length) as pbar:
            while True:
                pbar.update(1)
                if not class_data_local:
                    break

                if len(valid_list) > len(class_data[class_info[0]])*(1-train_ratio) or valid_sum > class_info[1]:
                    train_list.append(class_data_local[0])
                    for class_idx in range(len(class_data.keys())):
                        if class_data_local[0] in class_data[class_idx]:
                            class_data[class_idx].remove(class_data_local[0])
                    del class_data_local[0]
                else:
                    valid_list.append(class_data_local[0])
                    valid_sum += classified_data[class_data_local[0]][class_info[0]]
                    for class_idx in range(len(class_data.keys())):
                        if class_data_local[0] in class_data[class_idx]:
                            class_data[class_idx].remove(class_data_local[0])
                    del class_data_local[0]
        print(f'base_list length : {data_length}')
        print(f'train_list length : {len(train_list)}')
        print(f'valid_list length : {len(valid_list)}')

        return train_list, valid_list

    # main prosess
    def split_base_data(train_ratio=train_ratio):
        train_set, valid_set = init_dataset()
        sorted_data = sorting_data()
        classified_data, class_data = classify_image()

        train_list_total = []
        valid_list_total = []
        
        print(f'| data split start |')
        
        class_idx = 0
        while True:
            if class_idx > 9:
                break
            
            data_with_train_ratio = [[category, int(num*train_ratio)] for category, num in sorted_data]
            class_info = data_with_train_ratio[class_idx]

            train_list, valid_list = split_data_local(class_data, classified_data, class_info, train_ratio)
            train_list_total.extend(train_list)
            valid_list_total.extend(valid_list)

            class_idx += 1
        print('')
        print(f'| data split end |')
        print('')
        
        print(f'| make train_set start |')
        pbar = tqdm(train_list_total)
        for train_idx in pbar:
            for image in base_data_json['images']:
                if train_idx == image['id']:
                    train_set['images'].append(image)
                    break

            pre_state = 0
            state = 0
            for annot in base_data_json['annotations']:
                if pre_state - state == 1:
                    break
                if train_idx == annot['image_id']:
                    train_set['annotations'].append(annot)
                    pre_state = state
                    state = 1
                else:
                    pre_state = state
                    state = 0
        pbar.close()   
        print(f'| make train_set end |')
        print('')
        print(f'| make valid_set start |')
        pbar = tqdm(valid_list_total)
        for valid_idx in pbar:
            for image in base_data_json['images']:
                if valid_idx == image['id']:
                    valid_set['images'].append(image)
                    break

            pre_state = 0
            state = 0
            for annot in base_data_json['annotations']:
                if pre_state - state == 1:
                    break
                if valid_idx == annot['image_id']:
                    valid_set['annotations'].append(annot)
                    pre_state = state
                    state = 1
                else:
                    pre_state = state
                    state = 0
        pbar.close()
        print(f'| make valid_set end |')
        print('')

        return train_set, valid_set
    
    base_data_json = load_json()
    train_set, valid_set = split_base_data()

    with open('/opt/ml/dataset/train_set.json', 'w') as f:
        json.dump(train_set, f, indent=4)
    with open('/opt/ml/dataset/valid_set.json', 'w') as f:
        json.dump(valid_set, f, indent=4)
    
    print('| process success |')