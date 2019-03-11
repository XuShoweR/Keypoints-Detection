import os
import config
import pandas as pd

PATH_LOCAL = '/media/roots/data/keypoints_detection/Datasets/local_test'

df = pd.DataFrame(columns=['image_id', 'image_category'])
img_class_path = os.path.join(PATH_LOCAL, 'Images')
cur_record = 0
for class_dir in os.listdir(img_class_path):
    img_path_dir = os.path.join(img_class_path, class_dir)
    for img_path in os.listdir(img_path_dir):
        temp_content = 'Images/{}/{}'.format(class_dir, img_path)
        df.loc[cur_record] = [temp_content, class_dir]
        cur_record += 1
df.to_csv('{}/test.csv'.format(PATH_LOCAL), encoding='utf-8', index=False)