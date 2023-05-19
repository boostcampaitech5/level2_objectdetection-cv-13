import shutil
import json

src = '/opt/ml/dataset/'
src_train = '/opt/ml/dataset/train/'
dst_train_set = '/opt/ml/dataset/train_set/'
dst_valid_set = '/opt/ml/dataset/valid_set/'

for label_file_name in ['train_set.json', 'valid_set.json']:
    with open ('/opt/ml/dataset/' + label_file_name, 'r') as f:
        json_data = json.load(f)
        for image_info in json_data["images"]:
            image_file_name = image_info['file_name'].split('/')[1]
            if label_file_name == 'train_set.json':
                print(dst_train_set + image_file_name)
                shutil.copy(src_train + image_file_name, dst_train_set + image_file_name)
            elif label_file_name == 'valid_set.json':
                print(dst_valid_set + image_file_name)
                shutil.copy(src_train + image_file_name, dst_valid_set + image_file_name)
        f.close()
